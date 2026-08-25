import pyrealsense2 as rs
import numpy as np
import cv2
import csv
from datetime import datetime

# Configuration
marker_id = 2
marker_length = 0.14  # meters (14 cm)
filename = "marker_pose_spring.csv"

# Start RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Get intrinsics from RealSense
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(intrinsics.coeffs)

# ArUco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters_create()

print(f"Detecting ArUco marker ID {marker_id} (size: {marker_length*100:.1f} cm)...")
print("Press 's' to save the current marker pose, or 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None and marker_id in ids:
            idx = list(ids.flatten()).index(marker_id)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([corners[idx]], marker_length, camera_matrix, dist_coeffs)

            # Draw marker and axes
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.1)


            # Check key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Save current pose
                rvec = rvecs[0].ravel()
                tvec = tvecs[0].ravel()
                R, _ = cv2.Rodrigues(rvec)

                T_cam_marker = np.eye(4)
                T_cam_marker[:3, :3] = R
                T_cam_marker[:3, 3] = tvec

                timestamp = datetime.now().isoformat()
                with open(filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp] + T_cam_marker.flatten().tolist())

                print(f"Saved marker pose at {timestamp}")

            elif key == ord('q'):
                print("Quit pressed")
                break

        cv2.imshow("Marker Detection", image)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
