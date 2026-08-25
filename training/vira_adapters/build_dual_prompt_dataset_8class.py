"""Build the dual-prompt dataset that produced the deployed 8-class adapter.

Emits `augmented_camera1_crop2_multi_8class.json`, which is the dataset name
behind `...camera1-crop2-multi-8class-lora` - the checkpoint
vira_coarse_to_fine/servers/serve_dual_prompt.py loads.

build_dual_prompt_dataset.py is the sibling variant, writing
`augmented_camera1_crop2_multi.json`. Same four-turn structure; this is the one
whose output name matches the deployed adapter.

That adapter has since been deleted from the archive, so reproducing the
dual-prompt server means retraining from this dataset.
"""

import os
import json

# Define paths
base_path = os.path.join(BASE_DIR, "opentron_station15")
txt_file_path = os.path.join(base_path, "error_data.txt")
fail_image_folder = os.path.join(base_path, "color_images/camera1_renamed_cropped2")
fail_mapping_file = os.path.join(base_path, "color_images/image_row_mapping_camera1_renamed.json")
augmented_folder = os.path.join(base_path, "color_images/camera1_renamed_cropped2_augmented")
output_json = os.path.join(base_path, "augmented_camera1_crop2_multi_8class.json")

# Read movement data
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Movement classification (two-way only: no "No Move")
def classify_dx(dx):
    return "Move Down" if dx >= 0 else "Move Up"

def classify_dy(dy):
    return "Move Right" if dy >= 0 else "Move Left"

def classify_rotation(dtheta):
    return "Rotate Clockwise" if dtheta >= 0 else "Rotate Counterclockwise"

# Granularity (coarse/fine only)
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
                            "What movement (Move Up, Move Down, Move Left, Move Right) and what rotation "
                            "(Rotate Clockwise or Rotate Counterclockwise) are needed to align it properly?"
                        )
                    },
                    {
                        "from": "gpt",
                        "value": direction_label
                    },
                    {
                        "from": "human",
                        "value": (
                            "For the vertical position, horizontal position, and rotation of the object, is the misalignment large (coarse) or small (fine)?"
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

print(f"\n8-class movement + coarse/fine dataset saved: {output_json} ")
