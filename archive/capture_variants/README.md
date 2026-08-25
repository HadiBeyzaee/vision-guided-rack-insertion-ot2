# Dataset capture, as run

`save_data_opentron*.py`, the capture scripts used to collect the labelled
offset datasets. They differ in which cameras are recorded, the
offset ranges commanded, and the output directory.

All follow the same convention: apply a known offset, capture, append one
`dx dy dtheta` row to `error_data.txt`. The maintained versions are in
`../../data_collection/`.

## Files (17)

- `save_data_opentron.py`
- `save_data_opentron2.py`
- `save_data_opentron3.py`
- `save_data_opentron4.py`
- `save_data_opentron5.py`
- `save_data_opentron55.py`
- `save_data_opentron6.py`
- `save_data_opentron7.py`
- `save_data_opentron8.py`
- `save_data_opentron9.py`
- `save_data_opentron_black.py`
- `save_data_opentron_small_rack.py`
- `save_data_opentron_wrong_grasp.py`
- `save_data_opentron_wrong_grasp2.py`
- `save_data_opentron_wrong_grasp3.py`
- `save_data_opentron_wrong_grasp4.py`
- `save_data_opentron_wrong_grasp5.py`

**These move a real robot.** Paths and connection settings are
parameterised; the code is otherwise as it ran.
