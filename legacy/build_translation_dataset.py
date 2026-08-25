"""Build a translation-only dataset for the per-axis VLM variant.

Asks about Up/Down/Left/Right alone, with rotation handled by a second model.
Belongs to legacy/insert_with_llava_per_axis.py, which is the pre-OT-2 rig and
not part of either study.

Note its dead-band is 0.8 mm, not the 0.5 mm used everywhere else in this
repository. Do not merge its output into a dataset.
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
base_path = os.path.join(BASE_DIR, "opentron_stiff1")
txt_file_path = os.path.join(base_path, "error_data.txt")
fail_image_folder = os.path.join(base_path, "color_images/camera2_cropped_renamed")
fail_mapping_file = os.path.join(base_path, "color_images/image_row_mapping_camera2_renamed.json")
augmented_folder = os.path.join(base_path, "color_images/camera2_cropped_renamed_augmented")
output_json = os.path.join(base_path, "stiff_augmented_dxy.json")

# Read movement data
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Define translation-only labels
movement_x = ["Move Left", "Move Right", "No Move"]
movement_y = ["Move Up", "Move Down", "No Move"]
translation_labels = [f"{x}, {y}" for x in movement_x for y in movement_y]

# Function to classify dx (horizontal)
def classify_dx(dx):
    if abs(dx) < 0.0008:
        return "No Move"
    elif dx > 0.0008:
        return "Move Left"
    else:
        return "Move Right"

# Function to classify dy (vertical)
def classify_dy(dy):
    if abs(dy) < 0.0008:
        return "No Move"
    elif dy > 0.0008:
        return "Move Down"
    else:
        return "Move Up"

# Process augmented images
def process_augmented_images(mapping_file, augmented_folder, dataset):
    with open(mapping_file, "r") as f:
        image_mapping = json.load(f)

    for unique_id, data in image_mapping.items:
        original_name = data["original_name"]
        new_name = data["new_name"]
        row_number = data["row_number"]

        base_name = os.path.splitext(new_name)[0]
        related_aug_images = [
            img for img in os.listdir(augmented_folder) if img.startswith(base_name)
        ]

        if not related_aug_images or row_number > len(movement_lines):
            continue

        try:
            values = list(map(float, movement_lines[row_number - 1].split))
            dx, dy = values[0], values[1]
        except Exception:
            continue

        dx_label = classify_dx(dx)
        dy_label = classify_dy(dy)
        combined_instruction = f"{dx_label}, {dy_label}"

        for aug_img in related_aug_images:
            aug_image_path = os.path.join(augmented_folder, aug_img)
            entry = {
                "id": os.path.splitext(aug_img)[0],
                "image": aug_image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nHow should the black rack (bottom) move to align with the silver holder above? Choose one translation (Up, Down, Left, Right), or No Move if already aligned."
                    },
                    {
                        "from": "gpt",
                        "value": combined_instruction
                    }
                ]
            }
            dataset.append(entry)

# Main execution
dataset = []
process_augmented_images(fail_mapping_file, augmented_folder, dataset)

with open(output_json, "w") as json_file:
    json.dump(dataset, json_file, indent=4)

output_json
