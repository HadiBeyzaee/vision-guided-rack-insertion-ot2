# Serving configurations

Each pairs one fine-tuned adapter with one prompt on one port. They differ in the adapter loaded, the exact `--query` wording, the label list matched against, and the JSON key returned. The prompt must match what the adapter was trained on, which is why there are so many.

The three that the ViRA control loop actually drives live in `../../vira_coarse_to_fine/servers/`; the single-stage pair is in `../../serving/`.

## Files (18)

- `server_amount.py`
- `server_cnn1.py`
- `server_cnn2.py`
- `server_cnn2_v2.py`
- `server_cnn3.py`
- `server_cnn4.py`
- `server_cnn5.py`
- `server_cnn_cf.py`
- `server_cnn_single.py`
- `server_cnn_single2.py`
- `server_cnn_works_new_camera.py`
- `server_together_opentron2.py`
- `server_together_opentron3.py`
- `server_together_opentron4.py`
- `server_together_opentron5.py`
- `server_together_opentron7.py`
- `server_together_opentron_coarse_fine.py`
- `server_vlm.py`

Archived variants: paths parameterised, otherwise unmodified and not re-tested.
See [`../README.md`](../README.md) for why this folder exists.
