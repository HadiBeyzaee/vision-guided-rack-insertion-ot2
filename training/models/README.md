# Model training and offline scoring

Nothing here touches the robot.

## Two CNN trainers, and why both exist

They build different classifier heads, and the two are **not**
weight-compatible:

| Script | Head | Notes |
| --- | --- | --- |
| `train_cnn_27class_as_deployed.py` | `25088 -> 1024 -> 512 -> 27` | Produced `cnn_camera1_crop2_aug_wrong_station123.pth`, the VGG-19 behind the 90.3 % and 83.1 % results. The filename is hard-coded at its `torch.save`. |
| `train_cnn_27class.py` | `25088 -> 1024 -> 27` | The head as The simpler variant is it. |

A third shape, `1024 -> 1024 -> 27`, appears in
`../../archive/inference_servers/server_cnn5.py`.
`../../serving/serve_cnn_27class.py` reads the geometry out of whatever
checkpoint it is given rather than assuming one.

Use the as-deployed script to reproduce the reported model; use the other if
you want the architecture stated.

## Files

| File | What it does |
| --- | --- |
| `train_cnn_27class_as_deployed.py` | The deployed VGG-19 trainer. |
| `train_cnn_27class.py` | The two-layer-head variant. |
| `train_cnn_coarse_fine.py` | The 8-class Coarse/Fine head (2 x 2 x 2). |
| `finetune_llava.py` | LoRA fine-tuning of LLaVA-1.5-7B through DeepSpeed. |
| `merge_lora_adapter.py` | Merge an adapter into the base weights. The servers do not need this. |
| `evaluate_cnn.py` | Per-class accuracy on held-out images, with a bar chart. |
| `evaluate_llava.py` | The same for the VLM, via `run_llava.py`. |

## What these scripts do not measure

Both evaluators report **classification accuracy on held-out images**. That is
not insertion success, and it cannot reproduce the reported success rates, which were scored
by hand from the recorded trials. See
[`../../docs/caveats.md`](../../docs/caveats.md).

Both CNN trainers also mishandle the transform on a shared `Subset`, in opposite
directions - one ends up training without augmentation, the other applies it to
validation and test as well. Left as found, because changing them changes the
reported numbers. See
[`../../docs/caveats.md`](../../docs/caveats.md).
