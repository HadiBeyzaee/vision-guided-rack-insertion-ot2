# Hardware setup

## Platform

- **Arm**: Franka Emika Panda, mounted on a manually positioned dockable
  trolley. Docking variation during evaluation was approximately
  ±3 cm in the deck plane and ±6° in yaw.
- **Instrument**: Opentrons OT-2 liquid handler. Six deck slots are reachable
  from the docked position; slots are addressed `slot1`-`slot6`.
- **Cameras**: two wrist-mounted Intel RealSense D435i units, published as
  `/camera1/camera1/...` and `/camera2/camera2/...` by the RealSense ROS 2
  driver, at 1280 x 720.
- **Gripper**: Franka Hand, driven through `panda_py.libfranka.Gripper`.

## Which camera each study uses

The wrist camera was repositioned between the two studies, and this is the root
cause of most of the incompatibilities between them.

| | ViRA | CASE system |
| --- | --- | --- |
| Topic | `/camera1/camera1/color/image_raw` | `/camera2/camera2/color/image_raw` |
| Mounting | front-facing, views the rack-slot interface towards the open front of the deck | closer to the end effector, viewing the deck from above |
| Chosen because | gives an uncluttered view for front-row targets, kept for back-row targets to hold the configuration constant | retains rack-slot boundaries under back-row clutter *and* gives SAM-6D a usable view of the trolley surface |

`camera1` was also the front camera used for the entire ViRA dataset collection,
training and physical evaluation.

## Labware

| Item | Role |
| --- | --- |
| Opentrons 300 µL tip rack | the single training labware for both studies |
| 96-well PCR plate | unseen geometry, both studies |
| Blue 48-well tube rack | unseen, CASE only - not official Opentrons labware, but SBS-footprint compatible |
| 12-well reagent reservoir | unseen, CASE only |

Rack identity is passed through the pipeline as a colour string  -
`black` (tip rack), `white` (PCR plate), `blue` (reservoir),
`transparent` (storage plate) - which is what `localise_and_grasp_sam6d.py`
sends to the pose server as the `rack` form field, and what the task list in
`run_end_to_end_tasks.py` contains.

## Calibration you must redo for a new cell

Three things in this repository are specific to the original setup and will move
the arm to the wrong place if reused unchanged:

1. **Hand-eye transform.** `T_cam_ee` is a hard-coded 4 x 4 matrix in
   `complete_system/localise_and_grasp_sam6d.py`. It is the `EE_T_C` term of
   the pose transform.
2. **Deck slot waypoints.** `complete_system/transfer_to_deck_slot.py` holds three
   joint-space midpoints plus one taught joint configuration per slot. The
   midpoints exist because a direct Cartesian move clips the deck wall - do not
   replace them with a straight-line move without re-checking clearance.
3. **Camera intrinsics.** `config/camera1_cam.json` feeds the SAM-6D pose
   server. Regenerate with `robot_utils/print_camera_intrinsics.py` after any
   resolution change; intrinsics differ per stream profile.

Approach and insertion heights are likewise taught constants: the pose server
grasp approaches at `z = 0.23 m` and closes at `z = 0.13 m`, and the alignment
loops hold `z = 0.253 m` (ViRA) or `z = 0.31 m` (CASE) before dropping by
`dz_drop`.

## Bring-up order

1. Start the RealSense ROS 2 driver; confirm both camera topics publish.
2. Start the SAM-6D pose server (port 6000, external repository).
3. Start the alignment server(s) - see `docs/inference_servers.md`.
4. Verify predictions on a live frame with
   `robot_utils/post_frames_to_inference_server.py`, which moves nothing.
5. Only then run a control script.
