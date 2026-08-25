"""Build the four-turn dataset asking direction and granularity in one pass.

One conversation, two questions: the correction direction, then whether each
axis is Coarse or Fine. This is the two-question coarse phase as described in
the coarse-to-fine study.

The deployed control loop instead queries two separately fine-tuned adapters on
ports 5091 and 5012. Both configurations were trained; evaluate_dual_prompt.py
scores a checkpoint from this one. Which produced the reported insertion numbers
cannot be settled from the surviving files.
"""

import os

# --- Paths (override in your shell or a .env file) ---------------------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
# -----------------------------------------------------------------------
import os
import json

# Define paths
base_path = os.path.join(BASE_DIR, "opentron_station32")
txt_file_path = os.path.join(base_path, "error_data.txt")
fail_image_folder = os.path.join(base_path, "color_images/camera1_renamed_cropped2")
fail_mapping_file = os.path.join(base_path, "color_images/image_row_mapping_camera1_renamed.json")
augmented_folder = os.path.join(base_path, "color_images/camera1_renamed_cropped2_augmented")
output_json = os.path.join(base_path, "augmented_camera1_crop2_multi.json")

# Read movement data
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Movement classification (directional)
def classify_dx(dx):
    if abs(dx) < 0.0004:
        return "No Move"
    elif dx > 0.0004:
        return "Move Down"
    else:
        return "Move Up"

def classify_dy(dy):
    if abs(dy) < 0.0004:
        return "No Move"
    elif dy > 0.0004:
        return "Move Right"
    else:
        return "Move Left"

def classify_rotation(dtheta):
    if -0.1 <= dtheta <= 0.1:
        return "No Rotate"
    elif dtheta > 0.1:
        return "Rotate Clockwise"
    else:
        return "Rotate Counterclockwise"

# Granularity (coarse/fine)
def classify_granularity(dx, dy, dtheta):
    offset_dx = "Coarse" if abs(dx) > 0.005 else "Fine"
    offset_dy = "Coarse" if abs(dy) > 0.005 else "Fine"
    offset_dtheta = "Coarse" if abs(dtheta) > 1.0 else "Fine"
    return f"{offset_dx}, {offset_dy}, {offset_dtheta}"

# Process augmented image dataset
def process_augmented_images(mapping_file, image_folder, augmented_folder, dataset):
    with open(mapping_file, "r") as f:
        image_mapping = json.load(f)

    for unique_id, data in image_mapping.items:
        new_name = data["new_name"]
        row_number = data["row_number"]

        original_base = os.path.splitext(new_name)[0]
        related_aug_images = [
            img for img in os.listdir(augmented_folder)
            if img.startswith(original_base)
        ]

        if row_number > len(movement_lines):
            print(f"Skipping {new_name}: No movement data for row {row_number}")
            continue

        values = list(map(float, movement_lines[row_number - 1].split))
        if len(values) < 3:
            print(f"Skipping {new_name}: Incomplete movement data")
            continue

        dx, dy, dtheta = values

        direction_label = f"{classify_dx(dx)}, {classify_dy(dy)}, {classify_rotation(dtheta)}"
        granularity_label = classify_granularity(dx, dy, dtheta)

        for aug_img in related_aug_images:
            aug_image_path = os.path.join(augmented_folder, aug_img)
            aug_id = os.path.splitext(aug_img)[0]

            entry = {
                "id": aug_id,
                "image": aug_image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\nThis cropped image shows the object position and orientation relative to the slot. "
                            "What movement (Move Up, Move Down, Move Left, Move Right, or No Move) and what rotation "
                            "(Rotate Clockwise, Rotate Counterclockwise, or No Rotate) are needed to align it properly?"
                        )
                    },
                    {
                        "from": "gpt",
                        "value": direction_label
                    },
                    {
                        "from": "human",
                        "value": (
                            "Is the misalignment coarse or fine for each of the following: dx, dy, and rotation?"
                        )
                    },
                    {
                        "from": "gpt",
                        "value": granularity_label
                    }
                ]
            }

            dataset.append(entry)

# Initialize and run
dataset = []
process_augmented_images(fail_mapping_file, fail_image_folder, augmented_folder, dataset)

# Save to JSON
with open(output_json, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"\nMixed coarse/fine + movement dataset saved: {output_json} ")
