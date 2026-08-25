# Trolley pipeline, as run

Revisions of the trolley sequence: pose estimation,
grasp, transfer to a deck slot, CNN correction, retreat. `trolley_end_to_end*.py`
are the orchestrators, importing the stages rather than calling them as
subprocesses.

The maintained equivalents are in `../../complete_system/` and
`../../reference_implementation/ot2_rack_insertion/`. These carry the taught
poses and hard-coded slot lists from the days they were run.

## Files (33)

- `trolley_cnn_test_data.py`
- `trolley_end_to_end.py`
- `trolley_end_to_end2.py`
- `trolley_end_to_end3.py`
- `trolley_end_to_end_recording.py`
- `trolley_end_to_end_recording2.py`
- `trolley_exit_opentron.py`
- `trolley_joints_move.py`
- `trolley_joints_movement_opentron.py`
- `trolley_joints_movement_opentron2.py`
- `trolley_move_to_pose1.py`
- `trolley_move_to_pose2.py`
- `trolley_move_to_pose2_force.py`
- `trolley_move_to_pose2_force2.py`
- `trolley_move_to_pose3.py`
- `trolley_move_to_pose4.py`
- `trolley_move_to_pose5.py`
- `trolley_move_to_pose6.py`
- `trolley_pick_predict_insert.py`
- `trolley_pick_table_top_slot.py`
- `trolley_pick_table_top_slot2.py`
- `trolley_pick_table_top_slot3.py`
- `trolley_pick_table_top_slot4.py`
- `trolley_pose_estimation_rack.py`
- `trolley_pose_estimation_rack2.py`
- `trolley_pose_estimation_rack_grasp.py`
- `trolley_pose_estimation_rack_grasp2.py`
- `trolley_record_realsense.py`
- `trolley_record_realsense2.py`
- `trolley_test.py`
- `trolley_test_cnn_classification_dxy.py`
- `trolley_test_cnn_classification_dxy2.py`
- `trolley_test_cnn_classification_dxy3.py`

**These move a real robot.** Paths and connection settings are
parameterised; the code is otherwise as it ran.
