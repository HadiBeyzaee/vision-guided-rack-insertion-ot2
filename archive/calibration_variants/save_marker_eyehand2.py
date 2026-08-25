import os

# --- Connection settings (override in your shell or a .env file) -------
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
# -----------------------------------------------------------------------
import pyrealsense2 as rs
import numpy as np
import cv2
import csv
from datetime import datetime
import panda_py
from panda_py import libfranka
from scipy.spatial.transform import Rotation as R

hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)
panda.set_default_behavior()
# Configuration
marker_id = 4
marker_length = 0.028  # meters (14 cm)
filename = "opentron.csv"

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

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_ARUCO_ORIGINAL
)
parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

print(f"Detecting ArUco marker ID {marker_id} (size: {marker_length*100:.1f} cm)...")
print("Press 's' to save the current marker pose, or 'q' to quit.")
frame_count = 0

T_cam_ee = np.array([[ 0.01234,  0.99895,  0.04404, -0.05],
                            [-0.99988,  0.01192,  0.0099,   0.034 ],
                            [ 0.00937, -0.04416,  0.99898 ,-0.01893],
                            [ 0.    ,   0.,       0. ,      1.     ]]   )  

# T_cam_ee = np.array([[ 0.03729, -0.99926,  0.0097 ,  0.03231],
# [ 0.9984,   0.03684, -0.04298,  0.05633],
# [ 0.04259 , 0.01129,  0.99903,  0.02208],
# [ 0.   ,    0.    ,   0.    ,   1.     ]]  )  

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    image = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco_detector.detectMarkers(gray)


    if ids is not None and marker_id in ids:
        idx = list(ids.flatten()).index(marker_id)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[idx]], marker_length, camera_matrix, dist_coeffs
        )

        rvec = rvecs[0].ravel()
        tvec = tvecs[0].ravel()

        # Draw visuals
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

        # --- Build T_cam_marker ---
        R_cam_marker, _ = cv2.Rodrigues(rvec)
        T_cam_marker = np.eye(4)
        T_cam_marker[:3, :3] = R_cam_marker
        T_cam_marker[:3, 3] = tvec

        # --- Marker -> target offset ---
        T_marker_target = np.eye(4)
        T_marker_target[1, 3] = 0.05    # +5 cm X
        T_marker_target[0, 3] = 0.088  # -8.8 cm Y
        T_marker_target[2, 3] = 0.0

        pose = panda.get_pose()
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = pose[:3, :3]
        T_base_ee[:3, 3] = pose[:3, 3]
        T_base_obj = T_base_ee 

        # --- Full chain ---
        T_cam_target   = T_cam_marker @ T_marker_target
        T_ee_target    = T_cam_ee @ T_cam_target

        T_base_target  = T_base_ee @ T_ee_target

        # Throttled printing
        frame_count += 1
        if frame_count % 10 == 0:
            R_base_target = T_base_target[:3, :3]
            t_base_target = T_base_target[:3, 3]
            t_base_target[2] = 0.23

            # Convert rotation matrix -> quaternion
            orientation_quaternion = R.from_matrix(R_base_target).as_quat()
            # quaternion format: [x, y, z, w]

            # Convert quaternion -> Euler angles (roll, pitch, yaw)
            r = R.from_quat(orientation_quaternion)
            euler_angles = r.as_euler('xyz', degrees=True)

            roll, pitch, yaw = euler_angles

            print("Target pose w.r.t BASE frame:")
            print(T_base_target)

            print(f"Position [m]: "
                f"x={t_base_target[0]:.3f}, "
                f"y={t_base_target[1]:.3f}, "
                f"z={t_base_target[2]:.3f}")

            print(f"Orientation quaternion [x y z w]: "
                f"{orientation_quaternion}")

            print(f"Orientation RPY [deg]: "
                f"roll={roll:.2f}, "
                f"pitch={pitch:.2f}, "
                f"yaw={yaw:.2f}")

            print("-" * 60)


    cv2.imshow("Marker Detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
