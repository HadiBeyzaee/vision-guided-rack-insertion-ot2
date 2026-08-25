# Training

Dataset construction, model training and offline evaluation. Nothing here
touches the robot.

| Folder | Contents |
| --- | --- |
| [`dataset/`](dataset/) | Everything that turns raw captures and an `error_data.txt` into a training set. |
| [`models/`](models/) | CNN training, LLaVA LoRA fine-tuning, and offline scoring. |
| [`vira_adapters/`](vira_adapters/) | The ViRA study's three adapters - each has its own prompt and its own label set. |

## Pipeline order

```
dataset/rename_images.py         raw/<i>.png -> renamed/<uuid>.png + mapping
dataset/crop_images.py           crop to the region the model sees
dataset/augment_images.py        colour jitter, grayscale, blur
dataset/build_llava_dataset.py   offsets -> 27-class strings -> conversation JSON
dataset/build_cnn_dataset.py     the same labels, flattened for the CNN
                                 |
models/train_cnn_27class.py      or  models/finetune_llava.py
models/evaluate_cnn.py           or  models/evaluate_llava.py
```

For the single-stage (CASE) dataset, `dataset/compute_effective_dx.py` must run
**before** the builders - that label file has four columns, not three.

Full detail: [`../docs/dataset_format.md`](../docs/dataset_format.md).

## `dataset/`

| File | What it does |
| --- | --- |
| [`rename_images.py`](dataset/rename_images.py) | UUID rename, `{original_filename, new_filename, row_index}` schema - for `build_llava_dataset.py`. |
| [`rename_images_rowmapping.py`](dataset/rename_images_rowmapping.py) | UUID rename, `{original_name, new_name, row_number}` schema - for everything in `vira_adapters/`. |
| [`crop_images.py`](dataset/crop_images.py) | Crop to the region the model sees. Holds the ViRA coarse and fine margins. |
| [`crop_images_topdown.py`](dataset/crop_images_topdown.py) | The top-down `camera2` rectangle used by the complete system. |
| [`crop_images_blur_roi.py`](dataset/crop_images_blur_roi.py) | Crop variant that blurs outside a sharp band, to force attention onto the boundary. |
| [`augment_images.py`](dataset/augment_images.py) | Colour jitter, grayscale, Gaussian blur. Label-preserving. |
| [`augment_images_albumentations.py`](dataset/augment_images_albumentations.py) | Harsher photometric recipe: brightness/contrast, gamma, CLAHE, solarize, invert. |
| [`augment_images_sobel.py`](dataset/augment_images_sobel.py) | Sobel edge-magnitude copies - colour and texture discarded, boundaries kept. |
| [`convert_to_gray_clahe.py`](dataset/convert_to_gray_clahe.py) | Deterministic grayscale + CLAHE preprocessing. |
| [`build_llava_dataset.py`](dataset/build_llava_dataset.py) | Offsets to 27-class strings; LLaVA conversation JSON. |
| [`build_cnn_dataset.py`](dataset/build_cnn_dataset.py) | Flatten that to `{image, label}` pairs. |
| [`compute_effective_dx.py`](dataset/compute_effective_dx.py) | Collapse the 4-column CASE label file to 3 columns (the effective-displacement step). |
| [`fix_yaw_offset_labels.py`](dataset/fix_yaw_offset_labels.py) | Add the -1.4° nominal-yaw correction back into a label file. |
| [`renumber_images.py`](dataset/renumber_images.py) | Copy captures to `1.png ... N.png` in sorted order. |
| [`make_aligned_labels.py`](dataset/make_aligned_labels.py) | Write zero labels for the no-correction subset. |
| [`merge_datasets.py`](dataset/merge_datasets.py) | Concatenate the per-slot datasets into one, with counts. |

## `models/`

| File | What it does |
| --- | --- |
| [`train_cnn_27class_as_deployed.py`](models/train_cnn_27class_as_deployed.py) | **The script that trained the deployed model.** Head `1024 → 512 → 27`. |
| [`train_cnn_27class.py`](models/train_cnn_27class.py) | The head as described it: `1024 → 27`. Not weight-compatible with the above. |
| [`train_cnn_coarse_fine.py`](models/train_cnn_coarse_fine.py) | The 8-class Coarse/Fine head (2 x 2 x 2). |
| [`finetune_llava.py`](models/finetune_llava.py) | LoRA fine-tuning of LLaVA-1.5-7B via DeepSpeed. |
| [`merge_lora_adapter.py`](models/merge_lora_adapter.py) | Merge an adapter into the base weights. Not needed by the servers. |
| [`evaluate_cnn.py`](models/evaluate_cnn.py) | Per-class accuracy on held-out images; writes a bar chart. |
| [`evaluate_llava.py`](models/evaluate_llava.py) | The same for the VLM, via `run_llava.py`. |

VGG-19 is the variant used for the end-to-end evaluations: highest
insertion success, and an image-to-command cycle under one second against the
VLM's 6-7 s.

## Which renamer, which head

Two decisions here will bite silently if you get them wrong.

**Renamer.** Two mapping schemas exist. `rename_images.py` feeds
`build_llava_dataset.py`; `rename_images_rowmapping.py` feeds everything in
`vira_adapters/`. The wrong pairing raises a `KeyError` inside the builder that
names nothing useful.

**Head shape.** The deployed VGG-19 was built `25088 → 1024 → 512 → 27`, but
The simpler variant is `1024 → 27`. Both trainers are here. The two are not
weight-compatible; `serving/serve_cnn_27class.py` detects which a checkpoint
carries and builds to match.

## Two things to know before trusting a run

- The evaluation scripts report **classification accuracy on held-out images**,
  not insertion success. They cannot reproduce the reported success rates, which were
  counted by hand from the physical trials.
- Both CNN training scripts mishandle the transform on a shared `Subset`, in
  opposite directions: `train_cnn_27class.py` ends up training *without*
  augmentation, and `train_cnn_coarse_fine.py` applies it to validation and test
  as well. Both are left as found, because changing them changes the reported
  numbers. See [`../docs/caveats.md`](../docs/caveats.md).
