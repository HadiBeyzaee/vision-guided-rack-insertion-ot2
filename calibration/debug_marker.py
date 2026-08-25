"""
debug_marker_pose_viewer.py

Purpose:
Real-time visualization and debugging of ArUco marker pose estimation using Intel RealSense.
Press 'q' to exit.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import math

# ==============================
# User Configuration
# ==============================
MARKER_ID = 2
MARKER_SIZE_M = 0.14  # 14 cm marker side length

# ==============================
# Helper Functions
# ==============================

def rvec_to_euler(rvec):
    """Convert rotation vector to Euler angles in degrees."""
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    else:
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = 0

    return np.degrees([roll, pitch, yaw])

# ==============================
# RealSense Setup
# ==============================
pipeline = rs.pipeline
config = rs.config
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

intrinsics = profile.get_stream(
    rs.stream.color).as_video_stream_profile.get_intrinsics

camera_matrix = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
])

dist_coeffs = np.array(intrinsics.coeffs)

# ==============================
# ArUco Dictionary + Detector
# ==============================
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters_create

print("Running marker pose viewer... Press 'q' to exit.")

# ==============================
# Main Loop
# ==============================
try:
    while True:
        frames = pipeline.wait_for_frames
        color_frame = frames.get_color_frame
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if ids is not None and MARKER_ID in ids:
            idx = list(ids.flatten).index(MARKER_ID)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]],
                MARKER_SIZE_M,
                camera_matrix,
                dist_coeffs
            )

            rvec = rvecs[0].ravel
            tvec = tvecs[0].ravel

            # Draw coordinate axes for visualization
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

            # Compute Euler angles
            roll, pitch, yaw = rvec_to_euler(rvec)

            print(f"Marker {MARKER_ID} Translation (m): "
                  f"x={tvec[0]:.3f}, y={tvec[1]:.3f}, z={tvec[2]:.3f}")

            print(f"Marker {MARKER_ID} Orientation (deg): "
                  f"roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}")

        cv2.imshow("ArUco Marker Pose Debug", frame)

        # Quit if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop
    cv2.destroyAllWindows

print("Closed RealSense stream.")
