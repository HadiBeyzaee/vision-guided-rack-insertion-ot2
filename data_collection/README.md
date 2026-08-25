# Data collection

On-robot collection of the labelled offset datasets. Every script here
**commands a real Panda immediately on execution.**

The convention is the same throughout: apply a known offset, capture, and append
one `dx dy dtheta` line to `error_data.txt`. Labels are therefore derived from
the commanded motion, never hand-annotated. The image index comes from the row
count of the label file, so an interrupted run resumes cleanly.

Layout and the full preparation pipeline: [`../docs/dataset_format.md`](../docs/dataset_format.md).

## Which script belongs to which study

| File | Study | Cameras | Offsets |
| --- | --- | --- | --- |
| `collect_offsets_dual_camera.py` | ViRA | camera1 + camera2 | ±13 mm, ±4° |
| `collect_grasp_and_place_offsets.py` | CASE | camera2 | grasp `dx ∈ U(0, 8 mm)`, then place `dxx, dyy ∈ ±[6,15] mm`, `dθ ∈ ±5°` |
| `collect_offsets_single_camera.py` | either | camera2 | ±12 mm, ±5° |

`collect_grasp_and_place_offsets.py` is the one that produces the
two-source error of the single-stage study - an offset applied *before* the grasp, which
survives gripper closure because the fingers close along `y` and cannot
compensate along `x`, plus a second offset applied *after* transfer above the
slot. It was previously named `collect_grasp_and_place_offsets.py`, which
misdescribed what it does.

## Supporting captures

| File | What it does |
| --- | --- |
| `capture_rgbd_for_pose_estimation.py` | Aligned RGB + depth pairs in the layout SAM-6D expects, with intrinsics alongside. |
| `collect_slot_to_slot_trials.py` | Slot-to-slot transfer with a pickup x-perturbation. |
| `capture_failure_frames.py` | Manual keypress capture of camera-2 frames when something goes wrong during a live run. |

## Before running

`base_dir` is set near the top of each collector and decides where everything
lands. The nominal poses are taught constants for the original cell - check them
against your setup first. See the safety note in the top-level README.
