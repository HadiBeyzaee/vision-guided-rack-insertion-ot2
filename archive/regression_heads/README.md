# Regression instead of classification

Predicts the offset as a continuous value rather than a discrete correction - heads ending in `Linear(..., 1)` and an MSE-style objective, at 100x250 or 250x250 input.

The final system predicts one of 27 discrete corrections and applies a fixed step instead. These show the alternative framing that was tried first.

## Files (8)

- `train_cnn_regression.py`
- `train_cnn_regression2.py`
- `train_cnn_regression3.py`
- `train_cnn_regression4.py`
- `train_regression.py`
- `train_regression1.py`
- `train_regression2.py`
- `train_regression_v2.py`

Archived variants: paths parameterised, otherwise unmodified and not re-tested.
See [`../README.md`](../README.md) for why this folder exists.
