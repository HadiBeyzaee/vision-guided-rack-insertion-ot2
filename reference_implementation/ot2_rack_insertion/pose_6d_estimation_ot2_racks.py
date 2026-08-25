import os
import requests
import threading
import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import panda_py
from panda_py import libfranka
import roboticstoolbox as rtb
import spatialmath as sm


def clamp_yaw_transform(T, yaw_range=(-90, 90)):
    """Clamp yaw in a 4x4 transform to improve IK feasibility."""
    t = T[:3, 3]
    R_original = T[:3, :3]

    # Flip if upside down
    if R_original[2, 2] < 0:
        R_flip = R.from_euler('x', 180, degrees=True).as_matrix()
        R_original = R_original @ R_flip

    roll, pitch, yaw = R.from_matrix(R_original).as_euler('xyz', degrees=True)
    yaw = ((yaw + 180) % 360) - 180

    if yaw < yaw_range[0]:
        yaw += 180
    elif yaw > yaw_range[1]:
        yaw -= 180

    R_fixed = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    T_new = np.eye(4)
    T_new[:3, :3] = R_fixed
    T_new[:3, 3] = t

    return T_new, yaw


def solve_ik(T, q0):
    """Helper IK solver for Panda using Robotics Toolbox."""
    try:
        robot = rtb.models.ETS.Panda()
    except Exception:
        robot = rtb.models.DH.Panda()

    # Full DOF solve
    sol = robot.ikine_LM(T, q0=q0, ilimit=300, slimit=150, tol=1e-6)
    if sol.success:
        return sol.q

    # Mask: ignore some orientation
    sol = robot.ikine_LM(T, q0=q0, mask=[1, 1, 1, 0, 0, 1],
                         ilimit=300, slimit=150, tol=1e-6)
    return sol.q if sol.success else None


def run_pose_estimation(rack_color="white"):
    """Capture frame, estimate pose, transform and move Panda to grasp."""

    # Config
    server_url = "http:...:6000/pose"
    rgb_path = "/tmp/rgb.png"
    depth_path = "/tmp/depth.png"
    camera_path = "/path/to/camera1_cam.json"
    hostname = PANDA_HOSTNAME
    serial_number = "147322070835"

    # Safe pose
    q_safe = [1.6245797046670103, -0.11402809741964566,
             -0.08876788540685361, -1.4589830255597422,
            -0.07862584180588267, 1.432335729651981, 0.7393831875382199]

    # Camera-to-EE transform (from calibration)
    T_cam_ee = np.array([
        [0.01234, 0.99895, 0.04404, -0.05845],
        [-0.99988, 0.01192, 0.00990, 0.02930],
        [0.00937, -0.04416, 0.99898, -0.01893],
        [0.0, 0.0, 0.0, 1.0]
    ])

    panda = panda_py.Panda(hostname)
    panda.set_default_behavior()
    panda.move_to_joint_position(q_safe, speed_factor=0.05)
    print("Moved to safe starting configuration")

    stop_event = None
    rec_thread = None
    video_out = None
    pipeline = None
    video_path = None

    try:
        # ------------------------------------
        # Capture color + depth for server
        # ------------------------------------
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)

        for _ in range(30):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Frame capture failed")

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        cv2.imwrite(rgb_path, color_image)
        cv2.imwrite(depth_path, depth_image)
        print("Captured RGB and depth")

        pipeline.stop()

        # ------------------------------------
        # Start high-res recording
        # ------------------------------------
        pipeline = rs.pipeline()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = "path/to/videos"
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"pose_estimation_{timestamp}.avi")

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_out = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))
        if not video_out.isOpened():
            raise RuntimeError("Could not open video file")

        stop_event = threading.Event()
        align_rec = rs.align(rs.stream.color)

        def record_loop():
            while not stop_event.is_set():
                f = pipeline.wait_for_frames()
                a = align_rec.process(f)
                cf = a.get_color_frame()
                if cf:
                    frame = np.asanyarray(cf.get_data())
                    video_out.write(frame)

        rec_thread = threading.Thread(target=record_loop, daemon=True)
        rec_thread.start()

        # ------------------------------------
        # Send to pose estimation server
        # ------------------------------------
        with open(rgb_path, "rb") as f1, open(depth_path, "rb") as f2, open(camera_path, "rb") as f3:
            resp = requests.post(server_url, files={"rgb": f1, "depth": f2, "camera": f3},
                                 data={"rack": rack_color})
        data = resp.json()

        if "T_obj_cam" not in data:
            raise RuntimeError("Pose not in server response")

        T_obj_cam = np.array(data["T_obj_cam"], dtype=float)
        T_obj_cam_clamped, yaw_deg = clamp_yaw_transform(T_obj_cam)
        print(f"Clamped camera yaw: {yaw_deg:.2f} deg")

        # ------------------------------------
        # Transform to base frame
        # ------------------------------------
        T_ee_obj = T_cam_ee @ T_obj_cam_clamped
        ee_pose = panda.get_pose()

        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = ee_pose[:3, :3]
        T_base_ee[:3, 3] = ee_pose[:3, 3]
        T_base_obj = T_base_ee @ T_ee_obj

        # ------------------------------------
        # Build goal pose
        # ------------------------------------
        rpy = R.from_matrix(T_base_obj[:3, :3]).as_euler('xyz', degrees=True)
        roll, pitch = -180.0, 0.0
        yaw = ((rpy[2] + 180) % 360) - 180
        if yaw < 0:
            yaw += 180

        rot = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
        T_clamped = np.eye(4)
        T_clamped[:3, :3] = rot
        T_clamped[:3, 3] = T_base_obj[:3, 3]
        T_clamped[2, 3] = 0.23  # above slot

        goal = sm.SE3.Rt(rot, T_clamped[:3, 3])

        # ------------------------------------
        # IK + execution
        # ------------------------------------
        q_start = np.array(panda.get_state().q, dtype=float)
        q_target = solve_ik(goal, q_start)
        if q_target is None:
            raise RuntimeError("IK failed")

        panda.move_to_joint_position(q_target, speed_factor=0.02)

        # ------------------------------------
        # Lower + grasp
        # ------------------------------------
        current_pos = panda.get_position()
        new_pos = current_pos.copy()
        new_pos[2] = 0.13
        panda.move_to_pose(new_pos, panda.get_orientation(), speed_factor=0.02)

        gripper = libfranka.Gripper(hostname)
        gripper.grasp(0.05, 0.02, 20, 0.04, 0.04)
        print("Grasp done")

    except Exception as e:
        print("Pipeline failed:", str(e))

    finally:
        if stop_event: stop_event.set()
        if rec_thread: rec_thread.join(timeout=2)
        if video_out: video_out.release()
        if video_path: print(f"Video saved: {video_path}")
        if pipeline:
            try: pipeline.stop()
            except: pass
        print("RealSense stopped")


if __name__ == "__main__":
    run_pose_estimation("white")
