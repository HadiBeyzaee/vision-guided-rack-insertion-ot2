# Augmentation recipes

Photometric variants. All are label-preserving: they change appearance, never rack-slot geometry, so the pose label still holds.

The recipes that ship are in `../../training/dataset/` - torchvision jitter, the Albumentations set, Sobel edges, and grayscale+CLAHE.

## Files (3)

- `augmentation_llava.py`
- `augmentation_opentron.py`
- `augmentation_opentron2.py`

Archived variants: paths parameterised, otherwise unmodified and not re-tested.
See [`../README.md`](../README.md) for why this folder exists.
