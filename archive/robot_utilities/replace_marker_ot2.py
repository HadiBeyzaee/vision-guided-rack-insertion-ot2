import os

# --- Archived variant. Connection settings and paths parameterised. ----
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")
# -----------------------------------------------------------------------


import numpy as np
import cv2
import requests
import io
import json


def get_pose_matrix(rgb_image, depth_image, camera_params,
                    server_url="http://100.68.52.97:4001",
                    prompt="rounded rectangle"):

    # Convert to bytes
    rgb_bytes = io.BytesIO()
    np.save(rgb_bytes, rgb_image)
    rgb_bytes.seek(0)

    depth_bytes = io.BytesIO()
    np.save(depth_bytes, depth_image)
    depth_bytes.seek(0)

    # Send request
    files = {
        'rgb': ('rgb.npy', rgb_bytes, 'application/octet-stream'),
        'depth': ('depth.npy', depth_bytes, 'application/octet-stream')
    }

    data = {
        'camera_params': json.dumps(camera_params),
        'prompt': prompt
    }

    response = requests.post(
        f"{server_url}/pose_numpy",
        files=files,
        data=data,
        timeout=30
    )

    if response.status_code != 200:
        raise Exception(f"Server error {response.status_code}: {response.text}")

    result = response.json()

    if not result['success']:
        raise Exception(f"Pose estimation failed: {result.get('error')}")

    T = np.array(result['T'])

    return T


import pyrealsense2 as rs
import numpy as np
import cv2

def get_aligned_realsense_data():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Warm up
    for _ in range(10):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        raise RuntimeError("Could not get aligned frames")

    # Convert to numpy
    depth = np.asanyarray(depth_frame.get_data())
    rgb = np.asanyarray(color_frame.get_data())
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Camera intrinsics
    intr = color_frame.profile.as_video_stream_profile().intrinsics

    camera_params = {
        'fx': intr.fx,
        'fy': intr.fy,
        'cx': intr.ppx,
        'cy': intr.ppy,
        'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
    }

    pipeline.stop()
    return rgb, depth, camera_params


if __name__ == "__main__":

    # Get live aligned data from RealSense
    rgb, depth, camera_params = get_aligned_realsense_data()

    # Get pose
    T = get_pose_matrix(rgb, depth, camera_params)

    print("Transformation matrix:")
    print(T)
    print(f"\nPosition: {T[:3, 3]}")
    print(f"Rotation:\n{T[:3, :3]}")
