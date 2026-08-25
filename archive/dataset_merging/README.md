# Merging and format conversion

The per-slot merges - collection ran one folder per OT-2 slot and configuration, concatenated afterwards - plus LLaVA-conversation-to-CNN-pairs conversion.

The generalised replacement is `../../training/dataset/merge_datasets.py`, which takes arguments instead of hard-coded file lists and prints per-file counts.

## Files (6)

- `cnn_json.py`
- `cnn_json_to_json.py`
- `cnn_json_to_json_v2.py`
- `merge_json2.py`
- `merge_json2_v2.py`
- `merge_json3.py`

Archived variants: paths parameterised, otherwise unmodified and not re-tested.
See [`../README.md`](../README.md) for why this folder exists.
