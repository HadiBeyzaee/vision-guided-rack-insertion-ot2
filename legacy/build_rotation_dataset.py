"""Build a rotation-only dataset for the per-axis VLM variant.

The rotation half of the split that build_translation_dataset.py covers for
translation. Belongs to the pre-OT-2 rig, not to either study.
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
# Define paths
base_path = os.path.join(BASE_DIR, "opentron_stiff1")
txt_file_path = os.path.join(base_path, "error_data.txt")
fail_image_folder = os.path.join(base_path, "color_images/camera2_cropped_renamed")
fail_mapping_file = os.path.join(base_path, "color_images/image_row_mapping_camera2_renamed.json")
augmented_folder = os.path.join(base_path, "color_images/camera2_cropped_renamed_augmented")
output_json = os.path.join(base_path, "stiff_augmented_dt.json")

# Read movement data
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Define rotation labels
rotation_labels = ["Rotate Clockwise", "Rotate Counterclockwise", "No Rotate"]

# Function to classify rotation only
def classify_rotation(dtheta):
    if -0.1 <= dtheta <= 0.1:
        return "No Rotate"
    elif dtheta > 0.1:
        return "Rotate Clockwise"
    else:
        return "Rotate Counterclockwise"

# Process augmented images for rotation-only
def process_augmented_images(mapping_file, image_folder, augmented_folder, dataset):
    with open(mapping_file, "r") as f:
        image_mapping = json.load(f)

    for unique_id, data in image_mapping.items:
        original_name = data["original_name"]
        new_name = data["new_name"]
        row_number = data["row_number"]

        # Find augmented versions of the image
        base_id = os.path.splitext(new_name)[0]
        related_aug_images = [img for img in os.listdir(augmented_folder) if img.startswith(base_id)]

        if not related_aug_images:
            print(f"No augmented images found for {new_name}")
            continue

        # Get ground truth dtheta
        if row_number > len(movement_lines):
            print(f"Skipping {original_name}: No movement data at row {row_number}")
            continue

        try:
            dtheta = float(movement_lines[row_number - 1].split[2])
            rotation_label = classify_rotation(dtheta)
        except:
            print(f"Invalid data for {original_name}")
            continue

        # Process all related augmented images
        for aug_img in related_aug_images:
            aug_image_path = os.path.join(augmented_folder, aug_img)

            entry = {
                "id": os.path.splitext(aug_img)[0],
                "image": aug_image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\nHow should the black rack (bottom) rotate to align with the silver holder above? "
                            "Choose from: Rotate Clockwise, Rotate Counterclockwise, or No Rotate."
                        )
                    },
                    {"from": "gpt", "value": rotation_label}
                ]
            }

            dataset.append(entry)

# Initialize dataset
dataset = []

# Process augmented images for rotation
process_augmented_images(fail_mapping_file, fail_image_folder, augmented_folder, dataset)

# Save dataset
with open(output_json, "w") as json_file:
    json.dump(dataset, json_file, indent=4)

print(f"\nRotation-only dataset saved: {output_json} ")
