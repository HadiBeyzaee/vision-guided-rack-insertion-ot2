# Coarse-to-fine VLM alignment, fixed grasp

The rack is grasped centrally from a taught position, carried above a target
slot, given a known in-plane offset, and then aligned by
a fine-tuned LLaVA-1.5-7B before insertion. The robot-OT-2 arrangement is fixed
throughout, so this isolates the visual alignment stage.

Uses **`camera1`** (the front wrist camera). Do not mix these scripts with
`complete_system/` - the sign convention for `Move Up` is inverted there.

## The two phases

The loop queries three servers, once per second:

```
  COARSE PHASE                          FINE PHASE
  crop1 (440,80,380,230)                crop2 (450,140,460,310)
  :5091  direction    ---+              :5000  direction
  :5012  Coarse/Fine  ---+              steps 0.9 mm, 0.1 deg
  steps 7 mm, 0.8 deg
```

The coarse phase runs until port 5012 reports `Fine` on all three axes; the loop
then latches to the fine server for the rest of the run. Alignment ends on
`No Move, No Move, No Rotate`, after which `z` drops by 75 mm to seat the rack.

Two phases exist because inference costs 6-7 s per query - aligning a 2 cm
initial offset with 1 mm steps alone would take far too long.

## Files

| File | What it does |
| --- | --- |
| [`align_coarse_to_fine_llava.py`](align_coarse_to_fine_llava.py) | The study. Two-phase VLM alignment with adaptive step size, annotated session video. |
| [`align_single_stage_cnn.py`](align_single_stage_cnn.py) | The same harness with a CNN behind the endpoint (port 4002) instead of LLaVA. |
| [`servers/`](servers/) | The three LLaVA servers this loop expects, plus `serve_dual_prompt.py` - the single-adapter alternative to the coarse pair. |

## Running

Start the three servers (`docs/inference_servers.md`), confirm both cameras
publish, then:

```bash
python3 vira/align_coarse_to_fine_llava.py
```

**This commands the robot immediately.** There is no confirmation prompt.
The script writes an annotated AVI to `videos/`, so that directory must exist.
