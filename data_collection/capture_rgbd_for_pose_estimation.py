"""Capture aligned RGB and depth frames in the layout SAM-6D expects.

Writes numbered frame_XXXX pairs alongside the fixed camera intrinsics, ready
to hand to the 6-DoF pose pipeline. Kept separate from the other collectors
because the directory layout is dictated by SAM-6D, not by this project.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import json
import subprocess
import datetime
import datetime

# === CONFIG ===
cad_path = "rack_new3.ply"
template_path = "my_outputs_rack_new3/templates"
live_output_dir = os.path.join(os.path.dirname(template_path), "live")
cam_path = "camera_cam1.json"  # Use fixed camera intrinsics file
os.makedirs(live_output_dir, exist_ok=True)

# === INIT REALSENSE ===
pipeline = rs.pipeline
config = rs.config
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
print("RealSense camera started.")


frame_index = 1  # Start from frame_0001

try:
    while True:
        print(f"\nCapturing and processing frame {frame_index:04d}...")
        start = time.time

        # Warm-up
        for _ in range(30):
            pipeline.wait_for_frames

        frames = pipeline.wait_for_frames
        color_frame = frames.get_color_frame
        depth_frame = frames.get_depth_frame
        if not color_frame or not depth_frame:
            continue

        frame_dir = os.path.join(live_output_dir, f"frame_{frame_index:04d}")
        os.makedirs(frame_dir, exist_ok=True)
        rgb_path   = os.path.join(frame_dir, "rgb.png")
        depth_path = os.path.join(frame_dir, "depth.png")
        seg_path   = os.path.join(frame_dir, "detection.json")
        pose_vis_path = os.path.join(frame_dir, "pose.png")
        vis_path = os.path.join(frame_dir, "seg.png")

        color_image = np.asanyarray(color_frame.get_data)
        depth_image = np.asanyarray(depth_frame.get_data)
        cv2.imwrite(rgb_path, color_image)
        cv2.imwrite(depth_path, depth_image)

        subprocess.run([
            "python", "Instance_Segmentation_Model/run_inference_custom3.py",
            "--segmentor_model", "sam",
            "--output_dir", live_output_dir,
            "--cad_path", cad_path,
            "--rgb_path", rgb_path,
            "--depth_path", depth_path,
            "--cam_path", cam_path,
            "--template_path", template_path,
            "--out_json_name", seg_path,
            "--save_vis_path", vis_path
        ])

        subprocess.run([
            "python", "Pose_Estimation_Model/run_inference_custom3.py",
            "--model", "pose_estimation_model",
            "--config", "Pose_Estimation_Model/config/base.yaml",
            "--output_dir", live_output_dir,
            "--cad_path", cad_path,
            "--rgb_path", rgb_path,
            "--depth_path", depth_path,
            "--cam_path", cam_path,
            "--seg_path", seg_path,
            "--template_path", template_path,
            "--save_path", pose_vis_path
        ])

        print(f"Frame {frame_index:04d} processed in {time.time - start:.2f} seconds.")
        frame_index += 1
        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    pipeline.stop
