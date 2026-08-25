"""Build the 8-label Coarse/Fine dataset that drives the ViRA phase switch.

Labels each axis Coarse or Fine at the boundary where the control loop changes
step size:

    |dx| > 5 mm, |dy| > 5 mm, |dtheta| > 1.0 deg  ->  Coarse, else Fine

These are NOT the 0.5 mm / 0.2 deg dead-band that defines No Move in the
direction datasets. Conflating the two produces a model that switches to fine
steps far too early.

Feeds the adapter served by vira_coarse_to_fine/servers/serve_coarse_fine_flags.py.
"""

import os

# --- Paths (override in your shell or a .env file) ---------------------
# BASE_DIR       : dataset root holding error_data.txt and color_images/
# LLAVA_REPO     : checkout of haotian-liu/LLaVA (provides llava/eval/run_llava.py)
# LLAVA_BASE     : base llava-v1.5-7b weights
# CHECKPOINT_DIR : directory holding the fine-tuned LoRA adapters
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_REPO     = os.environ.get("LLAVA_REPO", "/opt/LLaVA")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
RUN_LLAVA      = os.path.join(LLAVA_REPO, "llava/eval/run_llava.py")
# -----------------------------------------------------------------------
import os
import json

# Define paths
txt_file_path =     os.path.join(BASE_DIR, "opentron_station11/error_data.txt")
fail_image_folder = os.path.join(BASE_DIR, "opentron_station11/color_images/camera1_renamed_crop3")
output_json =       os.path.join(BASE_DIR, "opentron_station11/camera1_crop3_123_coarse_fine.json")

# Read movement data from TXT file
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Coarse/fine classifier
def classify_combined(dx, dy, dtheta):
    offset_dx = "Coarse" if abs(dx) > 0.005 else "Fine"
    offset_dy = "Coarse" if abs(dy) > 0.005 else "Fine"
    offset_dtheta = "Coarse" if abs(dtheta) > 1.0 else "Fine"
    return f"{offset_dx}, {offset_dy}, {offset_dtheta}"

# Initialize dataset
dataset = []

# Process all images in the folder
image_files = sorted(
    [f for f in os.listdir(fail_image_folder) if f.endswith(".png") or f.endswith(".jpg")],
    key=lambda x: int(os.path.splitext(x)[0])  # Sort numerically by filename
)

for image_file in image_files:
    image_id = os.path.splitext(image_file)[0]

    try:
        row_index = int(image_id) - 1  # 1.png -> row 0
    except ValueError:
        print(f"Skipping invalid filename: {image_file}")
        continue

    if row_index >= len(movement_lines):
        print(f"Skipping {image_file}: No corresponding row in error_data.txt")
        continue

    values = list(map(float, movement_lines[row_index].split))
    if len(values) < 3:
        print(f"Skipping {image_file}: Invalid line format in error_data.txt")
        continue

    dx, dy, dtheta = values[:3]
    combined_instruction = classify_combined(dx, dy, dtheta)

    entry = {
        "id": image_id,
        "image": os.path.join(fail_image_folder, image_file),
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image>\nThis cropped image shows the object position and orientation relative to the slot. "
                    "In each direction, is the misalignment coarse or fine?"
                )
            },
            {"from": "gpt", "value": combined_instruction}
        ]
    }

    dataset.append(entry)

# Save JSON
with open(output_json, "w") as json_file:
    json.dump(dataset, json_file, indent=4)

print(f"\nLLaVA classification dataset saved: {output_json} ")
