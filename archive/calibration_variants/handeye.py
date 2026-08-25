# eye2hand_calibration.py
import numpy as np
import pandas as pd
import cv2
import json

# ---------- config ----------
EE_CSV = "ee_poses.csv"
MARKER_CSV = "marker_pose.csv"
SELECT_METHOD = "DANIILIDIS"   # one of: "TSAI", "PARK", "HORAUD", "DANIILIDIS"
OUT_JSON = "handeye_result.json"
np.set_printoptions(precision=5, suppress=True)

# ---------- helpers ----------
def load_poses_from_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    df = df[df.columns[:17]]  # keep first 17 cols (timestamp + 16 values)
    matrix_data = df.iloc[:, 1:17]
    if matrix_data.shape[1] != 16:
        raise ValueError(f"Row must be 1 timestamp + 16 values; got {matrix_data.shape[1]}")
    data = matrix_data.to_numpy(dtype=np.float64)
    poses = data.reshape((-1, 4, 4))
    return poses

def to_R_t_lists(poses):
    """Return lists of 3x3 R and 3x1 t as OpenCV expects."""
    R_list, t_list = [], []
    for T in poses:
        R_list.append(T[:3, :3].astype(np.float64))
        t_list.append(T[:3, 3].reshape(3, 1).astype(np.float64))
    return R_list, t_list

def solve_all_methods(Rg, tg, Rc, tc):
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    sols = {}
    for name, m in methods.items():
        Rc2ee, tc2ee = cv2.calibrateHandEye(Rg, tg, Rc, tc, method=m)
        T_cam2ee = np.eye(4, dtype=np.float64)
        T_cam2ee[:3, :3] = Rc2ee
        T_cam2ee[:3, 3] = tc2ee.reshape(3)
        T_ee2cam = np.linalg.inv(T_cam2ee)
        sols[name] = (T_ee2cam, T_cam2ee)
    return sols

def rot_to_angle_deg(R):
    # angle from rotation matrix
    tr = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))

def transform_diff(Ta, Tb):
    """Return (angle_deg, trans_mm) difference between two SE3 transforms."""
    Ra, ta = Ta[:3, :3], Ta[:3, 3]
    Rb, tb = Tb[:3, :3], Tb[:3, 3]
    Rdelta = Rb @ Ra.T
    angle = rot_to_angle_deg(Rdelta)
    trans_mm = np.linalg.norm(tb - ta) * 1000.0
    return angle, trans_mm

# ---------- main ----------
def main():
    print("Loading end-effector (robot) poses…")
    ee_poses = load_poses_from_csv(EE_CSV)

    print("Loading marker (camera) poses…")
    marker_poses = load_poses_from_csv(MARKER_CSV)

    if len(ee_poses) != len(marker_poses):
        raise ValueError(f"Pose count mismatch: {len(ee_poses)} EE vs {len(marker_poses)} marker")

    # Build lists for OpenCV
    R_gripper2base, t_gripper2base = to_R_t_lists(ee_poses)       # EE->Base
    R_target2cam,   t_target2cam   = to_R_t_lists(marker_poses)    # Marker->Cam

    print("\nRunning Eye-in-Hand Calibration (OpenCV) with all methods…")
    sols = solve_all_methods(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)

    # Show each method’s T_ee2cam
    baseline_name = "TSAI"
    baseline_T = sols[baseline_name][0]
    for name, (T_ee2cam, T_cam2ee) in sols.items():
        print(f"\n -  {name}  -   T_ee2cam (End-Effector -> Camera):")
        print(T_ee2cam)
        ang, mm = transform_diff(baseline_T, T_ee2cam)
        if name != baseline_name:
            print(f"Δ vs {baseline_name}: rotation ~{ang:.2f}°, translation ~{mm:.1f} mm")

    # Choose solution
    choose = SELECT_METHOD if SELECT_METHOD in sols else baseline_name
    T_ee2cam, T_cam2ee = sols[choose]

    print(f"\nChosen method: {choose}")
    #print("T_ee2cam:\n", T_ee2cam)
    print("T_cam2ee:\n", T_cam2ee)

    # Save
    out = {
        "method": choose,
        "T_ee2cam": T_ee2cam.tolist(),
        "T_cam2ee": T_cam2ee.tolist(),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {OUT_JSON}")

if __name__ == "__main__":
    main()
