# debug_marker_pose_viewer.py
import pyrealsense2 as rs
import numpy as np
import cv2
import math

marker_id = 2
marker_length = 0.14  # meters

# Start RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Get camera intrinsics
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(intrinsics.coeffs)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters_create()

def rvec_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])

print("Press 'q' to quit.")
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        if ids is not None and marker_id in ids:
            idx = list(ids.flatten()).index(marker_id)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([corners[idx]], marker_length, camera_matrix, dist_coeffs)

            rvec = rvecs[0].ravel()
            tvec = tvecs[0].ravel()
            euler = rvec_to_euler(rvec)

            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

            print(f"[Marker {marker_id}] Translation (m): x={tvec[0]:.3f}, y={tvec[1]:.3f}, z={tvec[2]:.3f}")
            print(f"[Marker {marker_id}] Orientation (deg): roll={euler[0]:.1f}, pitch={euler[1]:.1f}, yaw={euler[2]:.1f}")

        cv2.imshow("ArUco Marker Pose", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
