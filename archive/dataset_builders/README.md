# Labelling schemes

Every way the correction was posed as a supervised label:

- 27-class - the full direction set including the stop class
- 9-class - translation only, no rotation
- 16-label - direction with no stop class, for the coarse phase
- per-axis - translation and rotation asked separately
- yes/no - whether the offset is large, per axis
- reference-image - the misaligned image paired with an aligned one

Dead-bands vary between 0.0004 m and 0.0008 m, and yaw between 0.1° and 0.2°. One variant uses a dead-band of zero. Datasets built with different thresholds must never be merged.

## Files (14)

- `build_dataset_27class_offset_grasp.py`
- `build_dataset_offset_grasp_base2.py`
- `build_dataset_offset_grasp_camera2_dxyt1.py`
- `build_dataset_offset_grasp_camera2_dxyt4.py`
- `build_dataset_offset_grasp_slot6.py`
- `classification_dxyt.py`
- `llava_compare.py`
- `llava_dtheta.py`
- `llava_dxy.py`
- `llava_second_offset_big_yes_no.py`
- `llava_twenty_seven_together.py`
- `llava_twenty_seven_together2.py`
- `llava_twenty_seven_together_multi.py`
- `llava_twenty_seven_together_multi2.py`

Archived variants: paths parameterised, otherwise unmodified and not re-tested.
See [`../README.md`](../README.md) for why this folder exists.
