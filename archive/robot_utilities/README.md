# Robot and camera utilities

The tools used constantly during the experiments: read the current
joint state, move to a taught configuration, nudge the end effector, check
reachability, back out of the deck, record a human demonstration.

`move_above_racks_*` position the wrist camera over the trolley holding area so
SAM-6D can observe it, `_video` variants recording as they go. `ik_solver_rtb.py`
wraps the roboticstoolbox IK used by the pose-estimation path.
`rack_client_robot.py` / `rack_server_robot.py` are the split-machine setup,
where perception runs on the GPU box and motion on the robot PC.

**These move a real robot immediately**, and their poses are taught for the
original cell. The three maintained equivalents are in `../../robot_utils/`.

## Files (21)

- `check_joints.py`
- `check_params.py`
- `exit_opentron.py`
- `hdemo.py`
- `hdemocheck.py`
- `ik_solver_rtb.py`
- `mnew_joint.py`
- `move_above_racks_6d_starts.py`
- `move_above_racks_6d_starts3.py`
- `move_above_racks_6d_starts_video.py`
- `move_above_racks_sam6d_video.py`
- `move_inside_ot2_above_target_slot.py`
- `move_to_joint.py`
- `move_to_pose.py`
- `move_up.py`
- `rack_client_robot.py`
- `rack_server_robot.py`
- `read_current_state.py`
- `record_demo.py`
- `replace_marker_ot2.py`
- `robot_joints.py`

Archived: connection settings and paths parameterised, otherwise unmodified and
not re-tested. See [`../README.md`](../README.md).
