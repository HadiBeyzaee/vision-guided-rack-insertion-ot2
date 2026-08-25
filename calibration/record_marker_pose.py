import pyrealsense2 as rs
import numpy as np
import cv2
import csv
from datetime import datetime

# =========================================================
# User Configuration
# =========================================================
MARKER_ID = 2
MARKER_SIZE_M = 0.14  # Marker side length in meters
OUTPUT_CSV = "marker_pose_log.csv"

# =========================================================
# RealSense Camera Setup
# =========================================================
pipeline = rs.pipeline
config = rs.config
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Camera intrinsics
intr = profile.get_stream(rs.stream.color).as_video_stream_profile.get_intrinsics
camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0,       1     ]
])
dist_coeffs = np.array(intr.coeffs)

# =========================================================
# ArUco Detector Setup
# =========================================================
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters_create

print(f"Detecting ArUco marker ID {MARKER_ID} (size: {MARKER_SIZE_M * 100:.1f} cm)")
print("Press 's' to save current pose, 'q' to quit.")

# =========================================================
# Main Loop
# =========================================================
try:
    while True:
        frames = pipeline.wait_for_frames
        color_frame = frames.get_color_frame
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and MARKER_ID in ids.flatten:
            index = list(ids.flatten).index(MARKER_ID)
            pts = [corners[index]]

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                pts, MARKER_SIZE_M, camera_matrix, dist_coeffs
            )

            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.1)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Save pose
                rvec = rvecs[0].ravel
                tvec = tvecs[0].ravel
                R, _ = cv2.Rodrigues(rvec)

                T_cam_marker = np.eye(4)
                T_cam_marker[:3, :3] = R
                T_cam_marker[:3, 3] = tvec

                with open(OUTPUT_CSV, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now.isoformat] + T_cam_marker.flatten.tolist)

                print("Pose saved.")

            elif key == ord('q'):
                print("Quit requested.")
                break

        cv2.imshow("ArUco Marker Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # fallback exit trigger
            break

finally:
    pipeline.stop
    cv2.destroyAllWindows
