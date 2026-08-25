# Offline CNN scoring

The CNN counterparts to `../vlm_evaluation/`. Note the head shapes differ between them - some build `1024 → 512 → N`, others `1024 → N`. A script only loads checkpoints matching its own head.

## Files (6)

- `test_cnn_new.py`
- `test_cnn_new2.py`
- `test_cnn_new3.py`
- `test_cnn_new3_v2.py`
- `test_cnn_new_coarse_fine.py`
- `test_regression.py`

Archived variants: paths parameterised, otherwise unmodified and not re-tested.
See [`../README.md`](../README.md) for why this folder exists.
