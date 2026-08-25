# Complete end-to-end rack insertion

Autonomous marker-free localisation and grasping run in front of the alignment
stage, so the residual error the vision model has to correct now includes
pose-estimation error, hand-eye error, off-centre grasping and trolley-docking
variation.

Uses **`camera2`** (top-down wrist camera) and a single-stage 27-class
correction. `Move Up` maps to `position[0] -= step` here - the opposite of
`vira_coarse_to_fine/`.

## The pipeline

`run_end_to_end_tasks.py` is the orchestrator. For each `(rack, slot)` task:

```
1. localise_and_grasp_sam6d.run_pose_estimation(rack)   imported
2. transfer_to_deck_slot.move_to_slot(slot)                 imported
3. align_and_insert_cnn.py                        subprocess
4. retreat_from_deck.py                                 subprocess
```

Steps 3 and 4 run as subprocesses deliberately, so a failure in one task does
not end the whole run. Each task and the full sequence are timed.

Step 1 posts RGB, depth and `config/camera1_cam.json` to the SAM-6D pose server
on port 6000, clamps the returned yaw into the gripper's reachable range,
transforms into the base frame via `B_T_R = B_T_EE · EE_T_C · C_T_R`,
solves IK with `roboticstoolbox`, and grasps.

Step 3 crops to `{240, 320, 500, 200}`, posts to the CNN server on port 4001,
and applies the returned label as a 1.2 mm / 1.1 mm / 0.1° step. It requires
**three consecutive** `No Move, No Move, No Rotate` predictions before lowering,
with a duplicate-frame guard on the ROS header stamp.

## Files

| File | What it does |
| --- | --- |
| [`run_end_to_end_tasks.py`](run_end_to_end_tasks.py) | Orchestrator; runs a list of (rack, slot) tasks back to back. |
| [`localise_and_grasp_sam6d.py`](localise_and_grasp_sam6d.py) | Marker-free localisation and grasp via the SAM-6D pose server. |
| [`localise_and_grasp_sam6d_recorded.py`](localise_and_grasp_sam6d_recorded.py) | As above, recording the approach on video. |
| [`transfer_to_deck_slot.py`](transfer_to_deck_slot.py) | Carry the grasped rack to `slot1`-`slot6` via taught joint waypoints. |
| [`pick_rack_from_taught_pose.py`](pick_rack_from_taught_pose.py) | Fixed-position pick, for when pose estimation is not wanted. |
| [`align_and_insert_cnn.py`](align_and_insert_cnn.py) | Closed-loop 27-class refinement, then insert and release. |
| [`retreat_from_deck.py`](retreat_from_deck.py) | Lift clear of the slot walls before any lateral motion. |
| [`record_session_video.py`](record_session_video.py) | Record the RealSense colour stream on a background thread. |


## Running

```bash
export PANDA_HOSTNAME=... INFERENCE_HOST=... REALSENSE_SERIAL=...
# SAM-6D pose server on :6000, CNN server on :4001
python3 complete_system/run_end_to_end_tasks.py
```

Edit the `tasks` list at the top of the orchestrator first. **Commands the robot
immediately.**

Note that `align_and_insert_cnn.py` computes an Albumentations pipeline
and a Sobel edge image per frame and then discards both - it sends the plain
crop. See `docs/caveats.md` before assuming augmentation is applied at
inference time.
