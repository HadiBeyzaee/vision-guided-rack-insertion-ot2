# ViRA dataset builders and evaluators

The ViRA study (the coarse-to-fine study) trains **three separate LoRA adapters**, each with its
own prompt and its own label set. These are the scripts that build their
datasets and score them offline. They are kept apart from `../` because the
single-stage CASE study uses one adapter and one prompt.

| Adapter | Built by | Labels | Prompt | Served on |
| --- | --- | --- | --- | --- |
| Coarse direction | `build_direction_dataset_coarse.py` | **16** - no `No Move`, no `No Rotate` | "...What movement and rotation are needed to align it properly?" | 5091 |
| Coarse/Fine flags | `build_coarse_fine_dataset.py` | 8 - `Coarse`/`Fine` per axis | "In each direction, is the misalignment coarse or fine?" | 5012 |
| Fine direction | `build_direction_dataset_fine.py` | 27 - includes the stop class | "...what movement and what rotation are needed to align it properly?" | 5000 |

The coarse adapter deliberately has **no stop class**. During the coarse phase
the decision to stop belongs to the flag server, so the direction model is only
ever asked which way to go. This is why its `classify_movement` returns
`Move Down if dx >= 0 else Move Up` with no dead-band.

## Coarse / Fine thresholds

`build_coarse_fine_dataset.py` and `evaluate_coarse_fine.py` both use:

```
|dx| > 5 mm   -> Coarse    else Fine
|dy| > 5 mm   -> Coarse    else Fine
|dθ| > 1.0°   -> Coarse    else Fine
```

These are the boundary at which the control loop switches from 7 mm / 0.8° steps
to 0.9 mm / 0.1° steps. They are distinct from the 0.5 mm / 0.2° dead-band that
defines `No Move` / `No Rotate` in the direction datasets - do not conflate them.

## Files

| File | What it does |
| --- | --- |
| `build_direction_dataset_coarse.py` | 16-label direction dataset for the coarse adapter. |
| `build_coarse_fine_dataset.py` | 8-label Coarse/Fine dataset for the phase-switch adapter. |
| `build_direction_dataset_fine.py` | 27-label direction dataset, augmented copies included. |
| `evaluate_coarse_fine.py` | Offline accuracy of the flag adapter. |
| `build_dual_prompt_dataset.py` | Four-turn direction + granularity conversation. |
| `build_dual_prompt_dataset_8class.py` | The variant whose output name matches the deployed `...multi-8class-lora`. |
| `evaluate_direction.py` | Offline accuracy of a direction adapter. |

All of them read `BASE_DIR`, `LLAVA_REPO`, `LLAVA_BASE` and `CHECKPOINT_DIR` from
the environment. The prompt string in each evaluator **must** match the one its
server sends, or accuracy collapses at deployment while looking fine offline.

## Renamer

Every builder here reads the `{original_name, new_name, row_number}` mapping
schema, so they need `../dataset/rename_images_rowmapping.py`, **not**
`../dataset/rename_images.py`. See that file's docstring.
