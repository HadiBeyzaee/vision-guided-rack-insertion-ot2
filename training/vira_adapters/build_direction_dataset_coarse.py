"""Build the 16-label direction dataset for the ViRA coarse adapter.

Deliberately has NO "No Move" and NO "No Rotate" class, and no dead-band: the
label is simply the sign of each offset. During the coarse phase the decision to
stop belongs to the Coarse/Fine flag adapter, so this model is only ever asked
which way to go.

Feeds the adapter served by vira_coarse_to_fine/servers/serve_direction_coarse.py.
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
txt_file_path =     os.path.join(BASE_DIR, "opentron_station33/error_data.txt")
fail_image_folder = os.path.join(BASE_DIR, "opentron_station33/color_images/camera1_renamed_cropped3")
fail_mapping_file = os.path.join(BASE_DIR, "opentron_station33/color_images/image_row_mapping_camera1_renamed.json")
output_json =       os.path.join(BASE_DIR, "opentron_station33/camera1_crop3_nine_label.json")

# Read movement data from TXT file
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Movement classification (no "No Move")
def classify_movement(dx, dy):
    move_x = "Move Down" if dx >= 0 else "Move Up"
    move_y = "Move Right" if dy >= 0 else "Move Left"
    return move_x, move_y

# Rotation classification (no "No Rotate")
def classify_rotation(dtheta):
    return "Rotate Clockwise" if dtheta > 0 else "Rotate Counterclockwise"


# Function to generate full label
def classify_combined(dx, dy, dtheta):
    move_x, move_y = classify_movement(dx, dy)
    rotation_label = classify_rotation(dtheta)

    # Ensure all three labels exist, separate with commas
    return f"{move_x}, {move_y}, {rotation_label}".strip

# Function to process images
def process_images(mapping_file, image_folder, dataset):
    with open(mapping_file, "r") as f:
        image_mapping = json.load(f)

    for unique_id, data in image_mapping.items:
        original_name = data["original_name"]
        new_name = data["new_name"]
        row_number = data["row_number"]

        # Construct file path
        image_path = os.path.join(image_folder, new_name)

        # Get movement data from `error_data.txt`
        if row_number > len(movement_lines):
            print(f"Skipping {original_name}: No movement data for row {row_number}")
            continue

        movement_data = movement_lines[row_number - 1]  # Convert to 0-based index
        values = list(map(float, movement_data.split))
        if len(values) < 3:
            print(f"Skipping {original_name}: Invalid format in movement data")
            continue

        dx, dy, dtheta = values[0], values[1], values[2]

        # Generate combined instruction (ensuring three labels)
        combined_instruction = classify_combined(dx, dy, dtheta)

        # Structured LLaVA prompt with fixed format
        entry = {
            "id": unique_id,
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        "<image>\nThis cropped image shows the object position and orientation relative to the slot. "
                        "What movement and rotation are needed to align it properly?"

                    )

                },
                {"from": "gpt", "value": combined_instruction}
            ]
        }

        dataset.append(entry)

# Initialize dataset
dataset = []

# Process fail images
process_images(fail_mapping_file, fail_image_folder, dataset)

# Save dataset as a single JSON file
with open(output_json, "w") as json_file:
    json.dump(dataset, json_file, indent=4)

print(f"\nLLaVA classification dataset saved: {output_json} ")
