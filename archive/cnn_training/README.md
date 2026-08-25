# CNN training variants

Differ in head shape (`1024 → N` versus `1024 → 512 → N`), input size (336x336 for the OT-2 work, 100x250 and 250x100 for the earlier rig), class count (3, 8, 9, 27) and epoch budget.

The variant that produced the deployed model is kept in the main tree as `../../training/models/train_cnn_27class_as_deployed.py`.

## Files (9)

- `train_cnn3.py`
- `train_cnn_classification.py`
- `train_cnn_classification2.py`
- `train_cnn_classification3.py`
- `train_cnn_classification4.py`
- `train_cnn_classification5.py`
- `train_cnn_classification6.py`
- `train_cnn_classification8.py`
- `train_cnn_coarse_fine2.py`

Archived variants: paths parameterised, otherwise unmodified and not re-tested.
See [`../README.md`](../README.md) for why this folder exists.
