# Robot and camera utilities

Small helpers used while setting up or checking a cell. None of them run a control loop.

## Files

- `print_camera_intrinsics.py` - read the RealSense intrinsics that go into `config/camera1_cam.json`. Analysis only.
- `post_frames_to_inference_server.py` - forward a live camera topic to a prediction endpoint and log the replies, so you can check a newly deployed model without anything moving.
- `ik_joint_angles_from_pose.py` - solve joint angles for a Cartesian pose, to confirm it is reachable before committing to it. Reads robot state.
