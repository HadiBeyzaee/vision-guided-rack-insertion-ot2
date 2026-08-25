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


def run_pose_estimation(rack_color="white"):
    """
    Full SAM-6D pipeline:
    - Capture RGB + depth from RealSense @640x480 for server
    - Send to SAM-6D server for pose estimation
    - Restart RealSense @1280x720 for recording
    - Clamp pose yaw & reconstruct pose
    - Solve IK and move Panda
    - Grasp with gripper
    """

    # === CONFIG ===
    server_url = "http://138.253.127.95:6000/pose"
    rgb_path = "/tmp/rgb.png"
    depth_path = "/tmp/depth.png"
    camera_path = CAMERA_JSON
    hostname = PANDA_HOSTNAME
    serial_number = "147322070835"  # RealSense serial

    # === Panda setup ===
    panda = panda_py.Panda(hostname)
    panda.set_default_behavior

    # Move Panda to safe starting configuration (update with your values)
    q_safe = [1.6245797046670103, -0.11402809741964566, -0.08876788540685361, -1.4589830255597422,
               -0.07862584180588267, 1.432335729651981, 0.7393831875382199]
    panda.move_to_joint_position(q_safe, speed_factor=0.05)
    print("Panda moved to safe start configuration")

    # === RealSense setup for 640x480 (server capture) ===
    pipeline = rs.pipeline
    config = rs.config
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    ctx = rs.context
    found = False
    for d in ctx.devices:
        if d.get_info(rs.camera_info.serial_number) == serial_number:
            config.enable_device(serial_number)
            found = True
            break
    if not found:
        raise RuntimeError(f"RealSense device with serial {serial_number} not found.")

    pipeline.start(config)

    # Warm-up frames
    for _ in range(30):
        pipeline.wait_for_frames

    try:
        # === Capture one frame for server ===
        frames = pipeline.wait_for_frames
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame
        color_frame = aligned_frames.get_color_frame

        if not color_frame or not depth_frame:
            print("Frame capture failed.")
            return

        color_image = np.asanyarray(color_frame.get_data)
        depth_image = np.asanyarray(depth_frame.get_data)

        cv2.imwrite(rgb_path, color_image)
        cv2.imwrite(depth_path, depth_image)
        print(f"Saved RGB -> {rgb_path}, Depth -> {depth_path}")

        # === Stop pipeline before restart ===
        pipeline.stop
        print("Pipeline stopped after 640x480 capture")


        # # === Start video recording immediately after stopping 640x480 pipeline ===
        # pipeline = rs.pipeline
        # config = rs.config
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # ctx = rs.context
        # config.enable_device(serial_number)
        # pipeline.start(config)

        # timestamp = datetime.now.strftime("%Y%m%d_%H%M%S")
        # video_dir = os.path.join(BASE_DIR, "videos")
        # os.makedirs(video_dir, exist_ok=True)
        # video_path = os.path.join(video_dir, f"pose_estimation_{timestamp}.avi")

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # video_out = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))
        # if not video_out.isOpened:
        # raise RuntimeError("Could not open video writer")

        # print(f"Recording started -> {video_path}")

        # stop_event = threading.Event
        # align_rec = rs.align(rs.stream.color)

        # def record_loop:
        # while not stop_event.is_set:
        # frames = pipeline.wait_for_frames
        # aligned = align_rec.process(frames)
        # cf = aligned.get_color_frame
        # if cf:
        # frame = np.asanyarray(cf.get_data)
        # video_out.write(frame)

        # rec_thread = threading.Thread(target=record_loop, daemon=True)
        # rec_thread.start

        # === Now send to SAM-6D server ===
        print("Sending to pose estimation server...")
        files = {
            'rgb': open(rgb_path, 'rb'),
            'depth': open(depth_path, 'rb'),
            'camera': open(camera_path, 'rb')
        }
        data = {'rack': rack_color}
        response = requests.post(server_url, files=files, data=data)

        # Now parse server response
        response_json = response.json

        if "T_obj_cam" not in response_json:
            print("Pose not found in server response:", response_json)
            return

        # === Extract Pose ===
        T_obj_cam = np.array(response_json["T_obj_cam"])

        def rebuild_pose_with_clamped_yaw(T, yaw_range=(-90, 90)):
            t = T[:3, 3]
            R_original = T[:3, :3]
            if R_original[2, 2] < 0:  # flip if upside down
                R_flip = R.from_euler('x', 180, degrees=True).as_matrix
                R_original = R_original @ R_flip

            roll, pitch, yaw = R.from_matrix(R_original).as_euler('xyz', degrees=True)
            yaw = ((yaw + 180) % 360) - 180
            if yaw < yaw_range[0]:
                yaw += 180
            elif yaw > yaw_range[1]:
                yaw -= 180

            R_fixed = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix
            T_new = np.eye(4)
            T_new[:3, :3] = R_fixed
            T_new[:3, 3] = t
            return T_new, yaw

        T_obj_cam_clamped, yaw_deg = rebuild_pose_with_clamped_yaw(T_obj_cam)
        print("Pose clamped yaw (deg):", yaw_deg)


        # === Camera to EE transform (calibration) ===
        # T_cam_ee = np.array([[ 0.01684, -0.99885,  0.04483,-0.072],
        # [ 0.99984, 0.01655, -0.00688, -0.027],
        # [ 0.00613, 0.04494, 0.99897, -0.00064],
        # [ 0.    ,  0.  ,    0.  ,    1.     ]])

        T_cam_ee = np.array(        [[ 0.01234,  0.99895,  0.04404, -0.05845],
                                    [-0.99988,  0.01192,  0.0099,   0.0293 ],
                                    [ 0.00937, -0.04416,  0.99898,-0.01893],
                                    [ 0.   ,   0.,       0.,      1.     ]]   )

        T_ee_obj = T_cam_ee @ T_obj_cam_clamped

        # === Transform into robot base frame ===
        pose = panda.get_pose
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = pose[:3, :3]
        T_base_ee[:3, 3] = pose[:3, 3]
        T_base_obj = T_base_ee @ T_ee_obj
        print("Object pose in base frame:\n", T_base_obj)

        # === Orientation analysis ===
        rpy = R.from_matrix(T_base_obj[:3, :3]).as_euler('xyz', degrees=True)
        roll, pitch = -180, 0
        yaw = ((rpy[2] + 180) % 360) - 180
        if yaw < 0:
            yaw += 180
        print("Final RPY (deg):", [roll, pitch, yaw])

        rotmat = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix
        T_clamped = np.eye(4)
        T_clamped[:3, :3] = rotmat
        T_clamped[:3, 3] = T_base_obj[:3, 3]
        T_clamped[2, 3] = 0.23

        U, _, Vt = np.linalg.svd(T_clamped[:3, :3])
        Rfix = U @ Vt
        if np.linalg.det(Rfix) < 0:
            U[:, -1] *= -1
            Rfix = U @ Vt
        T_goal = sm.SE3.Rt(Rfix, T_clamped[:3, 3])

        # === IK solving ===
        try:
            robot = rtb.models.ETS.Panda
        except Exception:
            robot = rtb.models.DH.Panda
        q_start = np.array(panda.get_state.q, float)

        def solve_ik(T, q0):
            sol = robot.ikine_LM(T, q0=q0, ilimit=300, slimit=150, tol=1e-6)
            if sol.success: return sol.q
            sol = robot.ikine_LM(T, q0=q0, mask=[1, 1, 1, 0, 0, 1],
                                 ilimit=300, slimit=150, tol=1e-6)
            if sol.success: return sol.q
            return None

        q_target = solve_ik(T_goal, q_start)
        if q_target is None:
            raise RuntimeError("IK failed for T_goal.")
        panda.move_to_joint_position(q_target, speed_factor=0.02)
        print("Joint move completed")

        # === Lower and grasp ===
        current_position = panda.get_position
        new_position = current_position.copy
        new_position[2] = 0.13
        panda.move_to_pose(new_position, panda.get_orientation, speed_factor=0.02)

        gripper = libfranka.Gripper(hostname)
        gripper.grasp(0.05, 0.02, 20, 0.04, 0.04)
        print("Grasp executed")

    except Exception as e:
        print("Pipeline failed:", str(e))

    # finally:
    # try:
    # if 'stop_event' in locals:
    # stop_event.set
    # if 'rec_thread' in locals:
    # rec_thread.join(timeout=2)
    # if 'video_out' in locals:
    # video_out.release
    # print(f"Video saved -> {video_path}")
    # except Exception as e:
    # print(f"Cleanup warning: {e}")
    # try:
    # pipeline.stop
    # except Exception:
    # pass
    # print("RealSense pipeline stopped")
