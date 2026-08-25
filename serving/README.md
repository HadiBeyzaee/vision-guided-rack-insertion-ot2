# Inference servers

The two single-stage servers used by the CASE loop and the VLM comparison. Both accept a multipart POST with an `image` field on `/predict` and answer on the key `movement`. Ports, the JSON contract and how to start a full stack are documented in [`../docs/inference_servers.md`](../docs/inference_servers.md).

## Files

- `serve_cnn_27class.py` - VGG-19 or ResNet-18, 27-class softmax. `MODEL_TYPE`, `MODEL_PATH` and `PORT` come from the environment; defaults are VGG-19 on 4001.
- `serve_llava_27class.py` - LLaVA-1.5-7B, shelling out to `run_llava.py` in `LLAVA_REPO`. Defaults to 5000, which collides with the ViRA fine server; run one at a time or override `PORT`.
