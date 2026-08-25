# Vision-Guided Rack Insertion into the Opentrons OT-2

A Franka Emika Panda on a dockable trolley inserts standard labware into the
deck of an Opentrons OT-2 liquid handler.

SAM-6D estimates the rack pose without fiducial markers. Localisation, grasping
and transfer each leave residual in-plane error, so a wrist camera drives a
closed loop that corrects translation and yaw before the rack is lowered. Two
versions of that loop are included: coarse-to-fine with a fine-tuned
LLaVA-1.5-7B, and single-stage with a CNN.

Measured insertion success across four labware types and six deck slots: 90.3%
with a taught centred grasp, 83.1% end to end with SAM-6D localisation.

## Layout

| Folder | Contents |
| --- | --- |
| [`reference_implementation/`](reference_implementation/) | Tidied end-to-end system. Start here. |
| [`vira_coarse_to_fine/`](vira_coarse_to_fine/) | Coarse-to-fine VLM loop and its LLaVA servers. |
| [`complete_system/`](complete_system/) | SAM-6D localisation, grasp, transfer, single-stage alignment, insertion. |
| [`calibration/`](calibration/) | Eye-in-hand calibration for the camera-to-end-effector transform. |
| [`training/`](training/) | Dataset preparation, model training, offline scoring. |
| [`serving/`](serving/) | CNN and LLaVA inference servers. |
| [`data_collection/`](data_collection/) | On-robot collection of the labelled offset datasets. |
| [`robot_utils/`](robot_utils/) | IK check, camera intrinsics, live prediction probe. |
| [`config/`](config/) | Camera intrinsics for the SAM-6D pose server. |
| [`legacy/`](legacy/) | Earlier control variants from a previous rig. |
| [`archive/`](archive/) | Alternatives tried and set aside: other label schemes, prompt variants, regression heads, a Qwen port. |

Each top-level folder has its own README. `docs/` covers the server contract,
dataset format, hardware setup, and known caveats.

## The two loops are not interchangeable

They use different cameras, crops and step sizes, and **the same movement label
means opposite directions in each** (`Move Up` is `+x` in one and `-x` in the
other). Moving a model or a server between them without adjusting for that will
drive the arm the wrong way. Details in
[`vira_coarse_to_fine/README.md`](vira_coarse_to_fine/README.md) and
[`complete_system/README.md`](complete_system/README.md).

## Correction labels

Both loops predict one of 27 classes, the product of three per-axis decisions:

```
x:      Move Up          | Move Down               | No Move
y:      Move Left        | Move Right              | No Move
theta:  Rotate Clockwise | Rotate Counterclockwise | No Rotate
```

Emitted as a comma-separated string, for example
`"Move Down, Move Left, No Rotate"`. The CNN predicts a class directly; the VLM
generates the string as text and it is matched against the label list.
Alignment stops on `"No Move, No Move, No Rotate"`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then edit it
```

ROS 2 Humble provides `rclpy`, `cv_bridge` and `sensor_msgs`; do not install
those with pip. The robot scripts also need `panda_py` built against your
libfranka, and a RealSense ROS 2 driver publishing both wrist cameras.

## External dependencies

- **SAM-6D pose server**: `server_publisher_chem_racks_pose.py` in
  [SAM6D-Chemistry-Lab](https://github.com/HadiBeyzaee/SAM6D-Chemistry-Lab),
  serving `POST /pose` on port 6000. CAD templates are generated there.
- **LLaVA**: a checkout of
  [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA), providing
  `llava/eval/run_llava.py` and the DeepSpeed training entry point. Point
  `LLAVA_REPO` at it.


## Licence

MIT. See [LICENSE](LICENSE).
