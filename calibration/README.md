# Hand-eye calibration

Computes the camera-to-end-effector transform — the transform from camera frame to
end-effector frame that lets a SAM-6D pose estimate be expressed in the robot
base frame:

```
B_T_R = B_T_EE . EE_T_C . C_T_R
```

Eye-in-hand, via `cv2.calibrateHandEye` with the Tsai method, against an ArUco
marker of known size.

## Procedure

Run the two recorders together, moving the arm to a new pose between captures.
Each keypress saves one matched pair.

```bash
python3 record_ee_pose.py       # writes ee_pose_log.csv
python3 record_marker_pose.py   # writes marker_pose_log.csv, 's' to save
python3 eye2hand_calibration.py # reads both, prints T_cam2ee and T_ee2cam
```

Both CSVs must have the same number of rows and be in the same order, since the
solver pairs them by index. `debug_marker.py` shows the live detection if the
marker is not being picked up.

The resulting matrix goes into `complete_system/localise_and_grasp_sam6d.py`,
where it currently sits as a hard-coded `T_cam_ee` from the original cell.
**Recalibrate before running on any other setup** — an uncorrected hand-eye
error appears directly as a grasp offset, which is the failure mode the write-up
identifies as the main limitation.

## Files

| File | What it does |
| --- | --- |
| `record_ee_pose.py` | Log the end-effector pose as a 4x4 matrix per row. |
| `record_marker_pose.py` | Detect the ArUco marker and log the camera-to-marker pose. Set `MARKER_ID` and `MARKER_SIZE_M` first. |
| `eye2hand_calibration.py` | Solve for the camera-to-end-effector transform. |
| `debug_marker.py` | Live marker detection view, for when nothing is being detected. |

From the author's published `End_to_End_OT2_Rack_Insertion` repository.
