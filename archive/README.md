# Archive - the experimental record

The rest of the repository holds the configuration that produced the reported
results: one script per stage, the one the deployed system actually ran. This
folder holds **everything else that was built and tried** - 124 scripts covering
alternatives that were evaluated and set aside, ablations, the iterations between
them, and the real-robot trial code that was run on the day.

It is here because the finished pipeline hides the work. Reading only
`vira_coarse_to_fine/` and `complete_system/` you would not know that eight
prompt phrasings were compared, that a Qwen port was attempted, that the
alignment task was tried as regression before it was posed as classification, or
that the Coarse/Fine phase switch replaced an earlier three-way "how large is the
offset" head. That is the record of the research, and it is what these folders
preserve.

**Nothing here is required to run the system.** Paths are parameterised through
the same `BASE_DIR`, `LLAVA_REPO`, `LLAVA_BASE` and `CHECKPOINT_DIR` variables
used elsewhere, but these scripts are otherwise unmodified and were not
re-tested. Several reference checkpoints and dataset folders that no longer
exist.

## What is here

| Folder | Files | What it shows |
| --- | --- | --- |
| [`inference_servers/`](inference_servers/) | 18 | Every serving configuration tried: different adapters, prompts, ports and label sets. |
| [`vlm_evaluation/`](vlm_evaluation/) | 13 | Offline scoring of the VLM adapters - per-axis, nine-label, 27-label, coarse/fine, dual-prompt. |
| [`cnn_evaluation/`](cnn_evaluation/) | 6 | The same for the CNN heads. |
| [`cnn_training/`](cnn_training/) | 9 | CNN training variants: head shapes, input sizes, class counts. |
| [`lora_finetuning/`](lora_finetuning/) | 11 | LoRA launch configurations - rank, alpha, epochs, prompt set. |
| [`dataset_builders/`](dataset_builders/) | 14 | Every labelling scheme tried: 27-class, 9-class, per-axis, yes/no, reference-image. |
| [`dataset_merging/`](dataset_merging/) | 6 | The per-slot merges, and LLaVA→CNN dataset conversion. |
| [`cropping/`](cropping/) | 6 | Crop rectangle searches for both cameras. |
| [`augmentation/`](augmentation/) | 3 | Augmentation recipe variants. |
| [`offset_amount_branch/`](offset_amount_branch/) | 4 | The abandoned three-way "is the offset large" head, superseded by Coarse/Fine. |
| [`regression_heads/`](regression_heads/) | 8 | Predicting the offset as a continuous value instead of a class. |
| [`qwen_port/`](qwen_port/) | 1 | Converting the dataset to Qwen format. No Qwen result reached the write-up. |
| [`robot_utilities/`](robot_utilities/) | 21 | Real-robot tools: joint moves, state reads, IK, camera positioning for SAM-6D, the split client/server setup. |
| [`ot2_control/`](ot2_control/) | 4 | Driving the OT-2 liquid handler itself between insertion trials. |

## Threads worth knowing about

**Classification replaced regression.** `regression_heads/` predicts `dx`, `dy`
or `dtheta` as a continuous value. The final system predicts one of 27 discrete
corrections instead and applies a fixed step. The discrete form is what both
both studies here use.

**The phase switch changed shape.** `offset_amount_branch/` asks "is the offset
large in each direction: yes/no yes/no yes/no" with 7 mm and 1.0° thresholds.
That became the Coarse/Fine flag adapter at 5 mm and 1.0°, which is what ships.

**The label set was narrowed repeatedly.** `dataset_builders/` holds 27-class,
9-class translation-only, 16-label no-stop, per-axis, and yes/no variants, with
dead-bands from 0.0004 m to 0.0008 m - and one variant using zero, so nothing is
ever labelled `No Move`. See `../docs/caveats.md`, item 30.

**Prompt phrasing was tuned by hand.** The servers in `inference_servers/`
differ mainly in their `--query` string. The prompt must match what the adapter
was fine-tuned on; several of these exist because it did not.

## Chapter scope

Everything here is work: the vision-guided classification approach,
where a CNN or VLM predicts one of 27 discrete corrections.

The geometry-based line - edge and slot-line detection, spring-corner matching,
the two-stage geometric correction - belongs to a separate project and lives in the
`geometry-based-rack-alignment-ot2` repository. It is deliberately absent here.

That boundary is drawn by what a script's *alignment step* calls, not by what it
mentions. Several end-to-end runners use SAM-6D for localisation but then hand
off to the geometry-based edge method for final alignment; those are geometry-based runs and were excluded, along with anything importing
`main_marker_edge_*`, `recovery_*`, `step1_detect_marker_*` or `analyse_fyn`.

## Naming conventions in these files

The vocabulary is the author's working shorthand. It is worth knowing before
reading anything here:

| Term | Means |
| --- | --- |
| **wrong** | *wrong grasp* - a deliberately off-centre grasp, used to generate residual x error. It never meant the code was defective. Files carrying it have been renamed to `offset_grasp`; the original names are noted in their headers and in `../docs/archive_inventory.md`. |
| **align** | the no-correction subset: captures taken with the rack already seated, labelled `0.0 0.0 0.0`. |
| **station11 ... station33** | one capture folder per OT-2 slot and deck configuration, merged afterwards. |
| **nine** | a nine-label variant - translation only, no rotation. |
| **dxyt** | the `(dx, dy, dtheta)` offset triple. |
| **crop1 ... crop6** | successive crop rectangles from the search in `cropping/`. Numbering is chronological, not ordered by size. |
| trailing `2`, `3`, `_new`, `_final` | successive revisions. The highest number is not reliably the one that was used; check which checkpoint or dataset name a script writes. |
| trailing `_v2` | added here, not by the author: two drives held different files under one name, and both were kept. |

## A caution

Because the label thresholds, crops and prompts differ between variants, files
here are **not** interchangeable with the ones in the main tree, and datasets
built by different builders must not be merged. Treat this folder as a record,
not as a parts bin.

Full provenance for every file, including the ones not copied here, is in
[`../docs/archive_inventory.md`](../docs/archive_inventory.md).
