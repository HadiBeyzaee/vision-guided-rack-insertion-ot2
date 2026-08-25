# Dataset format

Both studies use the same on-disk convention: images numbered in capture order,
and one plain-text label file whose *row index* matches the image number.

## What the collectors write

```
<base_dir>/
  color_images/camera1/1.png, 2.png, ...
  color_images/camera2/1.png, ...
  grayscale_depth_images/cameraN/<i>.png
  color_mapped_depth_images/cameraN/<i>.png
  error_data.txt
```

`error_data.txt` holds one `dx dy dtheta` line per sample - the offset that was
*commanded*, in metres and degrees. Because the offset is known, labels are
derived arithmetically and no image is ever hand-annotated.

The image index is taken from the current row count of `error_data.txt`, so an
interrupted collection resumes cleanly: the label file is the source of truth.

`base_dir` is set at the top of each collector (`opentron`,
`camera2_dxyt4`, `presentation_images` in the archived copies).

## Offset ranges per collector

| Script | Cameras | Offsets applied |
| --- | --- | --- |
| `collect_offsets_dual_camera.py` | camera1 + camera2 | ±13 mm in x and y, ±4° yaw |
| `collect_offsets_single_camera.py` | camera2 | ±12 mm in x and y, ±5° yaw |
| `collect_grasp_and_place_offsets.py` | camera2 | grasp `dx ∈ U(0, 8 mm)`, then place `dxx, dyy ∈ ±[6, 15] mm`, `dθ ∈ ±5°` |

The third is the one that produces the two-source error of the single-stage study: an offset
applied *before* the grasp, which survives gripper closure along x, plus an
offset applied *after* transfer above the slot.

## Preparation pipeline

Run in this order - each step consumes the previous step's output.

```
1. rename_images.py       raw/<i>.png  ->  renamed/<uuid>.png
                          writes image_rename_mapping.json holding
                          {uuid: {original_filename, new_filename, row_index}}

2. crop_images.py         crop to the region the model sees
                          coarse {440, 80, 380, 230} or fine {450, 140, 460, 310}

3. augment_images.py  writes <base>_orig.png and <base>_augN.png
                          colour jitter p=0.8, grayscale p=0.3, blur p=0.3

4. build_llava_dataset.py joins the mapping to error_data.txt by row_index,
                          converts (dx, dy, dtheta) to a 27-class string,
                          emits augmented_llava_dataset.json

5. build_cnn_dataset.py   flattens that to [{image, label}, ...]
```

The UUID rename in step 1 matters: it decouples the filename from the row index
so that augmented copies of one capture (`<uuid>_orig`, `<uuid>_aug1`) all
resolve back to the same label without renumbering anything.

## LLaVA training JSON

```json
{
  "id": "3f2a..._aug1",
  "image": "/data/images/augmented/3f2a..._aug1.png",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat movement and rotation are needed to correctly align the object with its slot?"},
    {"from": "gpt",   "value": "Move Down, Move Left, No Rotate"}
  ]
}
```

The `human` turn must contain the `<image>` token. The question text here must
match the `--query` string the inference server sends, or accuracy collapses at
deployment even though training looked fine - the servers in `vira_coarse_to_fine/servers/`
each carry their own phase-specific prompt for exactly this reason.

## CNN training JSON

```json
{"image": "/data/images/augmented/3f2a..._aug1.png", "label": "Move Down, Move Left, No Rotate"}
```

`train_cnn_27class.py` silently skips any entry whose `label` is not one of the 27
strings, so a prompt or spelling drift shows up as a shrinking dataset rather
than an error. Check the reported sample count against your image count.

## Label thresholds

| Quantity | Dead-band | Positive | Negative |
| --- | --- | --- | --- |
| `dx` | < 0.5 mm → `No Move` | `Move Down` | `Move Up` |
| `dy` | < 0.5 mm → `No Move` | `Move Right` | `Move Left` |
| `dtheta` | ≤ 0.2° → `No Rotate` | `Rotate Clockwise` | `Rotate Counterclockwise` |

Defined in `build_llava_dataset.py` and repeated in `evaluate_cnn.py` and
`evaluate_llava.py`. They must agree across all three.
