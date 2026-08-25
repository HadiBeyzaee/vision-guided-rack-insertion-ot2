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
import spatialmath as sm
import subprocess
import time

# ===================== Helper Functions =====================
def solve_ik_external(T_goal, q_start):
    """Solve IK using external script."""
    python_bin = ROS_PYTHON
    script = IK_SOLVER

    args = (
        [python_bin, script]
        + T_goal.flatten().tolist()
        + q_start.tolist()
    )

    result = subprocess.run(
        list(map(str, args)),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"IK failed:\n{result.stderr}")

    if "IK_FAIL" in result.stdout:
        return None

    return np.array(list(map(float, result.stdout.split())))

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
    server_url = "http://100.68.52.97:6000/pose"
    rgb_path = "/tmp/rgb.png"
    depth_path = "/tmp/depth.png"
    camera_path = CAMERA_JSON
    hostname = PANDA_HOSTNAME
    serial_number = "147322070835"  # RealSense serial

    # === Panda setup ===
    panda = panda_py.Panda(hostname)
    panda.set_default_behavior()


    video_dir = "sam_6d_video"
    os.makedirs(video_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ============================================================
    # VIDEO 1: RECORD SAFE POSE MOVEMENT (1920x1080)
    # ============================================================
    print("\n VIDEO 1: Recording safe pose movement...")
    video1_path = os.path.join(video_dir, f"3_safe_pose_{rack_color}_{timestamp}.avi")

    # Start camera for video 1
    pipeline_vid1 = rs.pipeline()
    config_vid1 = rs.config()
    config_vid1.enable_device(serial_number)
    config_vid1.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipeline_vid1.start(config_vid1)

    # Warm up
    for _ in range(10):
        pipeline_vid1.wait_for_frames()

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video1_writer = cv2.VideoWriter(video1_path, fourcc, 30.0, (1920, 1080))

    # Recording thread for video 1
    stop_vid1 = threading.Event()

    def record_vid1():
        while not stop_vid1.is_set():
            try:
                frames = pipeline_vid1.wait_for_frames()
                color_frame = frames.get_color_frame()
                if color_frame:
                    image = np.asanyarray(color_frame.get_data())
                    video1_writer.write(image)
            except:
                pass

    vid1_thread = threading.Thread(target=record_vid1)
    vid1_thread.start()

    # Move Panda to safe starting configuration (update with your values)
    #q_safe = [1.6245797046670103, -0.11402809741964566, -0.08876788540685361, -1.4589830255597422,
               #-0.07862584180588267, 1.432335729651981, 0.7393831875382199]
    q_safe = [1.0756333809317202, -0.005765475120983626, 0.46165245322386417, -1.5925940179656946, -0.03788653025155812, 1.7214663156403436, 0.7895909227511314]

    panda.move_to_joint_position(q_safe, speed_factor=0.05)
    print("Panda moved to safe start configuration")

    # Stop video 1
    time.sleep(0.5)
    stop_vid1.set()
    vid1_thread.join()
    video1_writer.release()
    pipeline_vid1.stop()
    print(f"Video 1 saved: {video1_path}")

    # === RealSense setup for 640x480 (server capture) ===
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    ctx = rs.context()
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
        pipeline.wait_for_frames()

    try:
        # === Capture one frame for server ===
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not color_frame or not depth_frame:
            print("Frame capture failed.")
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        cv2.imwrite(rgb_path, color_image)
        cv2.imwrite(depth_path, depth_image)
        print(f"Saved RGB -> {rgb_path}, Depth -> {depth_path}")

        # === Stop pipeline before restart ===
        pipeline.stop()
        print("Pipeline stopped after 640x480 capture")

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
        response_json = response.json()

        if "T_obj_cam" not in response_json:
            print("Pose not found in server response:", response_json)
            return

        # === Extract Pose ===
        T_obj_cam = np.array(response_json["T_obj_cam"])

        def rebuild_pose_with_clamped_yaw(T, yaw_range=(-90, 90)):
            t = T[:3, 3]
            R_original = T[:3, :3]
            if R_original[2, 2] < 0:  # flip if upside down
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

        T_obj_cam_clamped, yaw_deg = rebuild_pose_with_clamped_yaw(T_obj_cam)
        print("Pose clamped yaw (deg):", yaw_deg)


        # === Camera to EE transform (calibration) ===
        # T_cam_ee = np.array([[ 0.01684, -0.99885,  0.04483 ,-0.072],
        # [ 0.99984 , 0.01655, -0.00688, -0.027],
        # [ 0.00613 , 0.04494 , 0.99897, -0.00064],
        # [ 0.     ,  0.   ,    0.   ,    1.     ]])

        T_cam_ee = np.array(        [[ 0.01234,  0.99895,  0.04404, -0.05845],
                                    [-0.99988,  0.01192,  0.0099,   0.0293 ],
                                    [ 0.00937, -0.04416,  0.99898 ,-0.01893],
                                    [ 0.    ,   0.,       0. ,      1.     ]]   )

        T_ee_obj = T_cam_ee @ T_obj_cam_clamped

        # === Transform into robot base frame ===
        pose = panda.get_pose()
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

        rotmat = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
        T_clamped = np.eye(4)
        T_clamped[:3, :3] = rotmat
        T_clamped[:3, 3] = T_base_obj[:3, 3]
        T_clamped[2, 3] = 0.23

        # ============================================================
        # VIDEO 2: RECORD GRASP SEQUENCE (1920x1080)
        # ============================================================
        print("\n VIDEO 2: Recording grasp sequence...")
        video2_path = os.path.join(video_dir, f"4_grasp_{rack_color}_{timestamp}.avi")

        # Start camera for video 2
        pipeline_vid2 = rs.pipeline()
        config_vid2 = rs.config()
        config_vid2.enable_device(serial_number)
        config_vid2.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        pipeline_vid2.start(config_vid2)

        # Warm up
        for _ in range(10):
            pipeline_vid2.wait_for_frames()

        # Setup video writer
        video2_writer = cv2.VideoWriter(video2_path, fourcc, 30.0, (1920, 1080))

        # Recording thread for video 2
        stop_vid2 = threading.Event()

        def record_vid2():
            while not stop_vid2.is_set():
                try:
                    frames = pipeline_vid2.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        image = np.asanyarray(color_frame.get_data())
                        video2_writer.write(image)
                except:
                    pass

        vid2_thread = threading.Thread(target=record_vid2)
        vid2_thread.start()


        q_start = np.array(panda.get_state().q, float)
        q_target = solve_ik_external(T_clamped, q_start)
        if q_target is None:
            raise RuntimeError("IK failed for T_goal.")
        panda.move_to_joint_position(q_target, speed_factor=0.02)
        print("Joint move completed")

        # === Lower and grasp ===
        current_position = panda.get_position()
        new_position = current_position.copy()
        new_position[2] = 0.13
        panda.move_to_pose(new_position, panda.get_orientation(), speed_factor=0.02)

        gripper = libfranka.Gripper(hostname)
        gripper.grasp(0.05, 0.02, 20, 0.04, 0.04)
        print("Grasp executed")

        # Stop video 2
        time.sleep(0.5)
        stop_vid2.set()
        vid2_thread.join()
        video2_writer.release()
        pipeline_vid2.stop()
        print(f"Video 2 saved: {video2_path}")

        print("\nAll videos saved successfully!")
        print(f"   Video 1 (Safe Pose): {video1_path}")
        print(f"   Video 2 (Grasp): {video2_path}")

    except Exception as e:
        print("Pipeline failed:", str(e))

    # finally:
    # try:
    # if 'stop_event' in locals():
    # stop_event.set()
    # if 'rec_thread' in locals():
    # rec_thread.join(timeout=2)
    # if 'video_out' in locals():
    # video_out.release()
    # print(f"Video saved -> {video_path}")
    # except Exception as e:
    # print(f"Cleanup warning: {e}")
    # try:
    # pipeline.stop()
    # except Exception:
    # pass
    # print("RealSense pipeline stopped")
