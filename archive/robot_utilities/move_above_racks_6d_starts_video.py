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
import json
import time
import subprocess

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


def segment_rack_via_server(rgb_path: str, rack_color: str, rack_name: str = None,
                            server_url: str = "http://100.68.52.97:5000/segment",
                            json_output_path: str = None):
    """
    Send RGB image to segmentation server and get back segmented RGB + JSON.
    Server will send progress updates to Slack.

    Args:
        rgb_path: Path to input RGB image
        rack_color: Color of rack to find ("blue", "white", "black", etc.)
        rack_name: Human-readable name (e.g., "Blue Tip Rack") for Slack messages
        server_url: URL of segmentation server
        json_output_path: Where to save the SAM-6D JSON (optional)

    Returns:
        dict with 'rgb' (numpy array) and 'json_path' (str), or None if failed
    """
    if rack_name is None:
        rack_name = f"{rack_color} rack"

    print(f"   Sending to rack segmentation server...")
    print(f"   Rack: {rack_name}")
    print(f"   Server: {server_url}")

    try:
        # Prepare request
        files = {'rgb': open(rgb_path, 'rb')}
        data = {
            'rack_color': rack_color,
            'rack_name': rack_name  # Server will use this in Slack messages
        }

        # Send request
        response = requests.post(server_url, files=files, data=data, timeout=60)

        if response.status_code != 200:
            print(f"Server error: {response.status_code}")
            try:
                error = response.json().get('error', 'Unknown error')
                print(f"   Error: {error}")
            except:
                pass
            return None

        # Decode image from response
        image_array = np.frombuffer(response.content, dtype=np.uint8)
        segmented_rgb = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if segmented_rgb is None:
            print("Failed to decode segmented image")
            return None

        print(f"Received segmented RGB ({segmented_rgb.shape})")

        # Get JSON from response headers
        json_content = response.headers.get('X-JSON-Content')
        json_path = response.headers.get('X-JSON-Path')

        if json_content:
            # Parse JSON
            json_data = json.loads(json_content)

            # Save JSON if output path provided
            if json_output_path:
                with open(json_output_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                print(f"SAM-6D JSON saved: {json_output_path}")
                final_json_path = json_output_path
            else:
                # Use temp path from server
                final_json_path = json_path
                print(f"SAM-6D JSON available: {json_path}")
        else:
            print("No JSON in response")
            final_json_path = None

        return {
            'rgb': segmented_rgb,
            'json_path': final_json_path,
            'json_data': json_data if json_content else None
        }

    except requests.exceptions.Timeout:
        print("Server timeout (60s)")
        return None
    except requests.exceptions.ConnectionError:
        print("Cannot connect to server")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_pose_estimation_with_segmentation(rack_color="blue", rack_real_name="Blue Tip Rack"):
    """
    Full SAM-6D pipeline:
    - Capture RGB + depth from RealSense @640x480 for server
    - Send to SAM-6D server for pose estimation
    - Record videos at 1920x1080
    - Clamp pose yaw & reconstruct pose
    - Solve IK and move Panda
    - Grasp with gripper
    """

    rgb_path = "/tmp/rgb.png"
    depth_path = "/tmp/depth.png"
    camera_path = CAMERA_JSON
    hostname = PANDA_HOSTNAME
    serial_number = "147322070835"  # RealSense serial

    # Video directory setup
    video_dir = "rack_replacement_videos"
    os.makedirs(video_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # === Panda setup ===
    panda = panda_py.Panda(hostname)
    panda.set_default_behavior()

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

    # Execute safe pose movement while recording
    print("Moving to safe starting configuration...")
    q_safe = [1.6245797046670103, -0.11402809741964566, -0.08876788540685361, -1.4589830255597422,
               -0.07862584180588267, 1.432335729651981, 0.7393831875382199]
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
            print(" Frame capture failed.")
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        cv2.imwrite(rgb_path, color_image)
        cv2.imwrite(depth_path, depth_image)
        print(f"Saved RGB -> {rgb_path}")

        # Stop 640x480 pipeline (we're done with it)
        pipeline.stop()
        print(" Stopped 640x480 capture pipeline")

        # Segmentation
        result = segment_rack_via_server(
            rgb_path=rgb_path,
            rack_color=rack_color,
            rack_name=rack_real_name,
            json_output_path=None
        )

        if result is None:
            print("Segmentation failed!")
            return None

        segmented_rgb = result['rgb']
        json_data = result['json_data']

        # Save JSON data locally
        local_json_path = f"/tmp/{rack_color}_rack_seg.json"
        with open(local_json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"JSON saved locally -> {local_json_path}")

        # Save segmented RGB
        segmented_rgb_path = "/tmp/rgb_segmented.png"
        cv2.imwrite(segmented_rgb_path, segmented_rgb)
        print(f"Segmented RGB saved -> {segmented_rgb_path}")

        # === STEP 2: SEND TO POSE ESTIMATION SERVER ===
        print("\nSTEP 2: Pose Estimation...")
        print(" Sending to pose estimation server...")

        files = {
            'rgb': open(rgb_path, 'rb'),
            'depth': open(depth_path, 'rb'),
            'seg_json': open(local_json_path, 'rb')
        }
        data = {'rack': rack_color}

        response = requests.post("http://100.68.52.97:6000/pose", files=files, data=data)

        response_json = response.json()

        if "T_obj_cam" not in response_json:
            print("Pose not found in server response:", response_json)
            return

        # === Extract Pose ===
        T_obj_cam = np.array(response_json["T_obj_cam"])

        def rebuild_pose_with_clamped_yaw(T, yaw_range=(-90, 90)):
            t = T[:3, 3]
            R_original = T[:3, :3]
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

        T_obj_cam_clamped, yaw_deg = rebuild_pose_with_clamped_yaw(T_obj_cam)
        print("Pose clamped yaw (deg):", yaw_deg)

        # Camera to EE transform (calibration)
        T_cam_ee = np.array([
            [ 0.01234,  0.99895,  0.04404, -0.05845],
            [-0.99988,  0.01192,  0.0099,   0.0293 ],
            [ 0.00937, -0.04416,  0.99898 ,-0.01893],
            [ 0.    ,   0.,       0. ,      1.     ]
        ])

        T_ee_obj = T_cam_ee @ T_obj_cam_clamped

        # Transform into robot base frame
        pose = panda.get_pose()
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = pose[:3, :3]
        T_base_ee[:3, 3] = pose[:3, 3]
        T_base_obj = T_base_ee @ T_ee_obj
        print(" Object pose in base frame:\n", T_base_obj)

        # Orientation analysis
        rpy = R.from_matrix(T_base_obj[:3, :3]).as_euler('xyz', degrees=True)
        roll, pitch = -180, 0
        yaw = ((rpy[2] + 180) % 360) - 180
        if yaw < 0:
            yaw += 180
        print(" Final RPY (deg):", [roll, pitch, yaw])

        rotmat = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
        T_clamped = np.eye(4)
        T_clamped[:3, :3] = rotmat
        T_clamped[:3, 3] = T_base_obj[:3, 3]
        T_clamped[2, 3] = 0.23

        q_start = np.array(panda.get_state().q, float)
        q_target = solve_ik_external(T_clamped, q_start)
        if q_target is None:
            raise RuntimeError("IK failed for T_goal.")

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

        # Execute grasp sequence while recording
        print("Moving to target pose...")
        panda.move_to_joint_position(q_target, speed_factor=0.02)

        print("Lowering to grasp position...")
        current_position = panda.get_position()
        new_position = current_position.copy()
        new_position[2] = 0.125
        panda.move_to_pose(new_position, panda.get_orientation(), speed_factor=0.02)

        print(" Executing grasp...")
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
        import traceback
        traceback.print_exc()
