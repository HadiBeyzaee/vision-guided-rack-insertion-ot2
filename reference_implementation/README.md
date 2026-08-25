# Reference implementation

A cleaned public release of the end-to-end system, from the
`End_to_End_OT2_Rack_Insertion` repository.

**This is the version to build on.** The rest of this repository preserves the
scripts as they were run for the write-up, which carry taught poses, commented-out
alternatives and duplicated configuration blocks. These are the same algorithms
written to be read and reused.

## What is different from `../complete_system/`

Same crop `{240, 320, 500, 200}`, same port 4001, same `movement` reply key,
same three-consecutive-`No Move` stopping rule, same sign convention. The
differences are structural:

| | `../complete_system/` (as run) | here (released) |
| --- | --- | --- |
| Composition | subprocess calls between stages | plain imports, one function per stage |
| Entry points | module-level code that moves the robot on import | `if __name__ == "__main__"` guards |
| Step sizes | 1.2 mm x, 1.1 mm y | 1.0 mm both — rounded during cleanup |
| Drop distance | 0.081 m (after a duplicated config block) | 0.08 m, defined once |
| Dead code | an Albumentations pipeline and Sobel pass, both computed then discarded | removed |
| Video | separate recorder scripts | one `RealSenseRecorder` reused across stages |

## Two runners, two experiments

| File | Experiment |
| --- | --- |
| `ot2_rack_insertion/main_end_to_end_6d_to_cnn.py` | the end-to-end evaluation — SAM-6D localisation, grasp, transfer, align, insert. The 83.1 % result. |
| `ot2_rack_insertion/main_end_to_end_fix_to_cnn.py` | the fixed-grasp evaluation — taught centred grasp with trolley-docking error only. The 90.3 % result. |

Both iterate a `TASKS` list of `(rack_colour, slot)` pairs and time each task.

## Layout

```
ot2_rack_insertion/   the pipeline: pose estimation, transfer, alignment,
                      insertion, retreat, recording, and the CNN server
vlm_cnn_train/        dataset preparation, CNN training, LoRA fine-tuning,
                      offline evaluation, and the inference servers
```

`vlm_cnn_train/` overlaps `../training/` and `../serving/` but adds the
two-question dual-prompt pair — `llava_dataset_generation_two_questions.py` and
`server_run_llava_two_questions.py` — which build and serve a single adapter
answering both the direction and the Coarse/Fine question.

## Before running

Connection settings come from `PANDA_HOSTNAME` and `INFERENCE_HOST`. Anything
left as `"CONFIGURE_ME"` was an absolute path on the author's machine and must
be set. Every deck pose is taught for the original cell, and
`../calibration/` must be rerun for any other setup.

**These move a real robot.**
