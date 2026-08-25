# Datasets and trained weights

None of this is in the repository, and none of it should be: the raw captures
alone are **542 GB**. This file records what exists, where it is, and what is
worth depositing.

## What exists

| Artefact | Volume | Location |
| --- | --- | --- |
| Raw captures (RGB + depth + colour-mapped depth, both cameras) | **301 GB** | `Windows1/opentron/` |
| Raw captures, second tranche | **241 GB** | `New Volume1/opentron/` |
| Cropped + augmented training images | **~23 GB** | `.../opentron_station*/color_images/*cropped*` |
| Dataset JSONs (45-49 files, three near-identical copies) | ~500 MB each | `opentron_codes/`, `opentron/` |
| Label files (`error_data*.txt`) | **2.5 MB, 70 files** | one per capture folder |
| LoRA adapters (~25) | ~80 GB as stored | `Windows1/opentron/checkpoints/` |
| CNN checkpoints | 47-177 MB each | `opentron/`, `opentron_codes/` |

The raw-to-trained ratio is severe: a 2,000-sample station folder holds ~36,000
raw PNGs (both cameras x colour, depth and colour-mapped depth, before and after
each move) but contributes only 2,000 cropped images to training.

## The capture folders

Named by what was varied, not by which study they belong to:

| Folder family | Samples | What it varies |
| --- | --- | --- |
| `opentron_station11` ... `station33` | 1,500-2,000 each | ViRA: one folder per slot and deck configuration, later merged |
| `slot5_dxyt_1/2/3` | 5,000 / 2,100 / 2,100 | Slot 5 offset sweeps - the write-up's stated collection slot |
| `camera2_dxyt1` ... `dxyt4` | 1,250-2,500 each | the top-down camera used by the complete system |
| `opentron_wrong_grasp_base1` ... `6` | 3,000-5,000 each | deliberate grasp offsets |
| `slot6_grasp_base_offset1` | 4,200 | grasp offset at a back-row slot |
| `opentron_both1` ... `6` | 3,000 each | both cameras recorded together |

`camera2_dxyt4` is the folder whose `error_data.txt` has **four columns** - the
grasp offset and the placement offset kept separate. See
`training/dataset/compute_effective_dx.py`.

## Trained weights the deployed system actually loaded

Everything the servers in this repository reference, and whether it survives:

| Server | Checkpoint | Weights | Status |
| --- | --- | --- | --- |
| `serve_direction_coarse.py` (5091) | `...camera1-crop2-nine-lora` | 346 MB | **present** |
| `serve_coarse_fine_flags.py` (5012) | `...camera1-crop1-coarse-fine-new2-lora` | 651 MB | **present** |
| `serve_direction_fine.py` (5000) | `...camera1-augmented-crop1-high-angle-epoch-twenty-lora` | 346 MB | **present** |
| `serve_dual_prompt.py` (5010) | `...camera1-crop2-multi-8class-lora` | - | **MISSING - deleted** |
| `serve_cnn_27class.py` (4001) | `cnn_camera1_crop2_aug_wrong_station123.pth` | 177 MB (VGG-19) | **present** |

Each adapter directory is stored at 3-5.7 GB, but only two files matter:
`adapter_model.safetensors` and `non_lora_trainables.bin`. The rest is an
intermediate `checkpoint-NNNNN/` and a 2 MB `trainer_state.json`. **Stripped,
the three ViRA adapters come to about 1.3 GB total** rather than 11.8 GB.

The dual-prompt adapter is gone, so `serve_dual_prompt.py` cannot be run as-is.
Its dataset builder and evaluator remain, so it can be retrained.

### Two CNN checkpoints, two different backbones

The archive holds both, at the same 27-class output:

- `cnn_camera1_crop2_aug_wrong_station123.pth` - **177 MB, VGG-19**. This is the
  backbone selected here, and the one behind the 90.3 % and 83.1 % results.
- `cnn_misalignment_classifier.pth` - **47 MB, ResNet-18**.

Size alone tells them apart. `serve_cnn_27class.py` now detects the backbone
from the checkpoint rather than trusting `MODEL_TYPE`, because loading one into
the other fails with unexpected-key errors that say nothing about the cause.

## What to deposit, and where

**Do not push any of this to GitHub.** The per-file cap is 100 MB and the repo
soft limit is around 1 GB; a single VGG-19 checkpoint breaches the first and one
adapter breaches the second. Git LFS on a free account will not hold it either.

A reasonable split:

| Where | What | Size |
| --- | --- | --- |
| **This repo** | Code, and optionally the 70 `error_data*.txt` label files | ~3 MB |
| **Zenodo / figshare** (mint a DOI, cite it from the write-up) | Cropped + augmented images for the final merged datasets, the merged JSONs, the four surviving checkpoints stripped to their weight files | ~25 GB |
| **Institutional archive or the drives** | Raw captures | 542 GB |

The label files are the highest-value-per-byte artefact in the whole archive:
2.5 MB that, together with the capture images, regenerate every dataset in the
project. If only one thing is deposited alongside the code, deposit those.

Before depositing, rewrite the absolute paths inside the JSONs - every `image`
field points at `/media/hadi/Windows/opentron/...`, which resolves on no other
machine.

## Reproducing a dataset from a capture folder

```bash
export BASE_DIR=/path/to/opentron_station11
python3 training/dataset/rename_images.py        # UUID names + row mapping
python3 training/dataset/crop_images.py          # coarse or fine rectangle
python3 training/dataset/augment_images.py       # jitter, grayscale, blur
python3 training/dataset/build_llava_dataset.py  # -> conversation JSON
python3 training/dataset/build_cnn_dataset.py    # -> {image, label} pairs
```

For a four-column label file, run `compute_effective_dx.py` first. To rebuild a
full training set, repeat per station folder and concatenate with
`merge_datasets.py`. Merge only files sharing the same prompt and label set.
