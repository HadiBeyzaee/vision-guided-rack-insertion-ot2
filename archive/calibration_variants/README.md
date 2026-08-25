# Calibration, as run

Earlier hand-eye calibration scripts, alongside the maintained
versions in `../../calibration/`. `handeye.py` and `calibration_tag*.py` are
earlier solver revisions; `save_marker_eyehand*.py` and `save_poses_eyehand.py`
record the matched pose pairs; `get_camera_parameters.py` reads the RealSense
intrinsics.

The recorded pose pairs and the solved transform they produced are kept in
`../../calibration/data/`.

## Files (7)

- `calibration_tag.py`
- `calibration_tag2.py`
- `get_camera_parameters.py`
- `handeye.py`
- `save_marker_eyehand.py`
- `save_marker_eyehand2.py`
- `save_poses_eyehand.py`

**These move a real robot.** Paths and connection settings are
parameterised; the code is otherwise as it ran.
