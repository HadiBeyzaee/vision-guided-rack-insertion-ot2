# Earlier VLM control variants

Kept for provenance. **These are not part of either study.** Both subscribe to `camera2` with crop `{84, 300, 242, 30}` and a 10-20 mm `dz_drop`, which is the earlier rack-holder rig, not the OT-2 deck. They were filed under the VLM folder by name; the camera topic and crop geometry place them elsewhere.

## Files

- `insert_with_llava_single_label.py` - insertion driven by a single VLM movement label. The baseline the other variants refine.
- `insert_with_llava_per_axis.py` - the VLM queried once per axis, against separate translation and rotation endpoints (5001 / 5002).

## Per-axis dataset builders

These build the datasets for the per-axis variant above - one model asked only
about translation, another only about rotation. Neither is used by either
study, which ask a single question covering all three axes.

- `build_translation_dataset.py` - translation-only labels. Note its dead-band is
  0.8 mm, not the 0.5 mm used everywhere else.
- `build_rotation_dataset.py` - rotation-only labels.

## Servers

`servers/` holds the two endpoints the per-axis loop needs, so this folder now
runs standalone rather than referring to servers that were not archived here.

- `serve_translation.py` - port 5001, route `/predict_translation`, key `movement`.
- `serve_rotation.py` - port 5002, route `/predict_rotation`, key `rotation`.
