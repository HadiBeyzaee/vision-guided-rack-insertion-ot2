# Inference servers: ports and JSON contract

Every alignment loop in this repository is a ROS 2 node that crops a wrist-camera
frame, POSTs it as multipart form field `image`, and reads a movement label back
out of the JSON reply. The robot never loads a model itself.

## The contract

Request: `POST <host>:<port>/predict`, multipart, one file field named `image`.

Reply: JSON. **Every client in this repository reads the key `movement`.**

```json
{ "movement": "Move Down, Move Left, No Rotate" }
```

The coarse/fine flag server is the one exception; it answers on the key
`movement_cf` with a `Coarse`/`Fine` triple.

Servers match the model's text output against a fixed label list by substring,
returning `"Unknown"` on no match. A client that receives `"Unknown"` logs a
warning and takes no step, so a prompt/label mismatch shows up as a loop that
never moves rather than as a crash.

## Port map

| Port | Server | Returns | Client |
| --- | --- | --- | --- |
| 5091 | `vira_coarse_to_fine/servers/serve_direction_coarse.py` | `movement` (16 labels) | `align_coarse_to_fine_llava.py`, coarse phase |
| 5012 | `vira_coarse_to_fine/servers/serve_coarse_fine_flags.py` | `movement_cf` | same, to decide when to switch phase |
| 5000 | `vira_coarse_to_fine/servers/serve_direction_fine.py` | `movement` (27 labels) | same, fine phase |
| 5010 | `vira_coarse_to_fine/servers/serve_dual_prompt.py` | `predicted_movement` + `predicted_granularity` | none - the single-adapter alternative to 5091 + 5012 |
| 5000 | `serving/serve_llava_27class.py` | `movement` | single-stage VLM path (the CNN-vs-VLM comparison) |
| 4001 | `serving/serve_cnn_27class.py` | `movement` | `complete_system/align_and_insert_cnn.py` |
| 4002 | `serving/serve_cnn_27class.py` with `PORT=4002` | `movement` | `vira_coarse_to_fine/align_single_stage_cnn.py` |
| 5001 | `legacy/servers/serve_translation.py` (`/predict_translation`) | `movement` | `legacy/insert_with_llava_per_axis.py` |
| 5002 | `legacy/servers/serve_rotation.py` (`/predict_rotation`) | `rotation` | same |
| 6000 | `server_publisher_chem_racks_pose.py` (external, `SAM6D-Chemistry-Lab`) | `T_obj_cam` | `complete_system/localise_and_grasp_sam6d.py` |

## The two ways to run the coarse phase

Both were built and trained, and the archive contains both:

- **Split adapters (what the control loop drives).** Two separately fine-tuned
  models on 5091 and 5012, one per question. Two adapters to train and hold in
  memory, one generation each per frame.
- **One dual-prompt adapter (port 5010).** A single model queried twice per
  frame with two prompts, answering both questions in one reply. One adapter,
  but two sequential generations per frame.

The dual-prompt server answers on `predicted_movement` and
`predicted_granularity`, **not** the `movement` / `movement_cf` the control loop
reads, so swapping it in needs a client change or a key rename.

`serving/serve_llava_27class.py` and `vira_coarse_to_fine/servers/serve_direction_fine.py` both
default to 5000. Run only one at a time, or override with `PORT`.

## The coarse-phase label set

`serve_direction_coarse.py` builds its label list from
`["Move Up", "Move Down"] x ["Move Left", "Move Right"] x [4 rotation spellings]`
 -  **16 labels, with no `No Move` or `No Rotate`**. That is deliberate: during the
coarse phase the decision to stop belongs to the coarse/fine flag server on port
5012, not to the direction server. If you repoint that client at a 27-class
server, the loop will terminate on the first `No Move` triple and skip the fine
phase entirely.

The fine server carries the full label set including `No Move` / `No Rotate`,
plus lower-case spelling variants (`"Rotate clockwise"`) that the fine-tuned
model sometimes emits.

## Starting a stack

ViRA, three GPUs' worth of adapters (or sequentially on one):

```bash
export LLAVA_REPO=/opt/LLaVA LLAVA_BASE=/data/llava/llava-v1.5-7b
export CHECKPOINT_DIR=/data/checkpoints
python vira_coarse_to_fine/servers/serve_direction_coarse.py     # 5091
python vira_coarse_to_fine/servers/serve_coarse_fine_flags.py    # 5012
python vira_coarse_to_fine/servers/serve_direction_fine.py       # 5000
```

CASE, one CNN server:

```bash
MODEL_TYPE=vgg MODEL_PATH=checkpoints/cnn_misalignment_classifier.pth \
PORT=4001 python serving/serve_cnn_27class.py
```

Check it before letting the arm move:

```bash
curl -X POST http://<host>:4001/predict -F "image=@a_cropped_frame.png"
```

`robot_utils/post_frames_to_inference_server.py` does the same against a live
camera topic without commanding any motion.
