"""
Compute Eye-in-Hand calibration using OpenCV's calibrateHandEye().
Requires matched samples of:
 - End-effector poses      (4x4 transformation matrices)
 - Marker / target poses   (4x4 transformation matrices)
"""

import numpy as np
import pandas as pd
import cv2
import os

# =========================================================
# User Configuration
# =========================================================

EE_POSES_CSV =     "ee_pose_log.csv"
MARKER_POSES_CSV = "marker_pose_log.csv"


# =========================================================
# Helpers
# =========================================================

def load_poses_from_csv(file_path):
    """Load timestamp + 4x4 matrix per row → return Nx4x4 pose array."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, header=None)
    if df.shape[1] < 17:
        raise ValueError(f"Invalid format: expected 1 timestamp + 16 pose values, got {df.shape[1]} columns")

    matrix_data = df.iloc[:, 1:17]  # skip timestamp
    data = matrix_data.to_numpy(dtype=np.float64)

    if data.shape[1] != 16:
        raise ValueError(f"Each row must have 16 matrix numbers, got {data.shape[1]}")

    return data.reshape(-1, 4, 4)


def print_matrix(name, T):
    """Simple formatted matrix printout."""
    print(f"\n{name}:")
    print(np.array2string(T, formatter={'float_kind':lambda x: f'{x: .5f}'}))


# =========================================================
# Main Calibration
# =========================================================

def run_eye_to_hand():
    print("Loading robot EE poses...")
    ee_poses = load_poses_from_csv(EE_POSES_CSV)

    print("Loading camera→marker poses...")
    marker_poses = load_poses_from_csv(MARKER_POSES_CSV)

    if len(ee_poses) != len(marker_poses):
        raise ValueError(
            f"Pose count mismatch: {len(ee_poses)} EE vs {len(marker_poses)} marker poses"
        )

    # Split into rotation + translation
    R_gripper2base = [pose[:3, :3] for pose in ee_poses]
    t_gripper2base = [pose[:3, 3] for pose in ee_poses]
    R_target2cam   = [pose[:3, :3] for pose in marker_poses]
    t_target2cam   = [pose[:3, 3] for pose in marker_poses]

    print("Running OpenCV calibrateHandEye...")
    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # Build 4x4
    T_cam2ee = np.eye(4)
    T_cam2ee[:3, :3] = R_cam2ee
    T_cam2ee[:3, 3] = t_cam2ee.flatten()

    # Inverse transform: EE → Camera
    T_ee2cam = np.linalg.inv(T_cam2ee)

    print_matrix("Camera → End-effector (T_cam2ee)", T_cam2ee)
    print_matrix("End-effector → Camera (T_ee2cam)", T_ee2cam)

    return T_ee2cam, T_cam2ee


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    run_eye_to_hand()
