"""Archived variant. Originally named `llava_wrong.py`.

"wrong" in the original name is shorthand for **wrong grasp** - the
deliberately off-centre grasp used to generate residual x error. It never
meant that the code was defective. Renamed here so a reader does not have to
guess.
"""

import os
import json

# # Define paths
# base_path = os.path.join(BASE_DIR, "slot6_grasp_base_offset1")
# txt_file_path = os.path.join(base_path, "error_data_new.txt")
# fail_image_folder = os.path.join(base_path, "color_images/camera1_renamed_cropped4")
# fail_mapping_file = os.path.join(base_path, "color_images/image_row_mapping_camera1_renamed.json")
# augmented_folder = os.path.join(base_path, "color_images/camera1_renamed_cropped4_augmented")
# output_json = os.path.join(base_path, "augmented_camera1_crop4.json")

# Define paths
base_path = os.path.join(BASE_DIR, "slot6_grasp_base_offset1")
txt_file_path = os.path.join(base_path, "error_data_align.txt")
fail_image_folder = os.path.join(base_path, "color_images/camera1_align_renamed_cropped4")
fail_mapping_file = os.path.join(base_path, "color_images/image_row_mapping_camera1_align_new_renamed.json")
augmented_folder = os.path.join(base_path, "color_images/camera1_align_renamed_cropped4_augmented")
output_json = os.path.join(base_path, "augmented_camera1_crop4_align.json")


# Read movement data from TXT file
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Define 27-class labels (dx, dy, dtheta-based)

movement_y = ["Move Left", "Move Right", "No Move"]
movement_x = ["Move Up", "Move Down", "No Move"]


# Ensure same order
combined_labels = [
    f"{x}, {y}"
    for x in movement_x
    for y in movement_y

]

# Function to classify dx movement (Up/Down)
def classify_dx(dx):
    if abs(dx) < 0.0004:
        return "No Move"
    elif dx > 0.0004:
        return "Move Down"
    else:
        return "Move Up"

# Function to classify dy movement (Left/Right)
def classify_dy(dy):
    if abs(dy) < 0.0004:
        return "No Move"
    elif dy > 0.0004:
        return "Move Right"
    else:
        return "Move Left"


# Function to process augmented images correctly
def process_augmented_images(mapping_file, image_folder, augmented_folder, dataset):
    with open(mapping_file, "r") as f:
        image_mapping = json.load(f)

    for unique_id, data in image_mapping.items:
        original_name = data["original_name"]  # Example: "1.png"
        new_name = data["new_name"]  # Example: "e7eb98419dc34aa0b01dc308b0042ff9.png"
        row_number = data["row_number"]  # Example: 1

        # Find all augmented images related to this unique image
        original_base = os.path.splitext(new_name)[0]  # Remove .png -> "e7eb98419dc34aa0b01dc308b0042ff9"
        related_aug_images = [
            img for img in os.listdir(augmented_folder) if img.startswith(original_base)
        ]

        if not related_aug_images:
            print(f"No augmented images found for {new_name}")
            continue

        # Get movement data from `error_data.txt`
        if row_number > len(movement_lines):
            print(f"Skipping {original_name}: No movement data for row {row_number}")
            continue

        movement_data = movement_lines[row_number - 1]  # Convert to 0-based index
        values = list(map(float, movement_data.split))
        if len(values) < 2:
            print(f"Skipping {original_name}: Invalid movement data format")
            continue

        dx, dy = values[0], values[1]

        # Generate combined label
        dx_label = classify_dx(dx)
        dy_label = classify_dy(dy)

        combined_instruction = f"{dx_label}, {dy_label}"

        # Process each augmented version of the original image
        for aug_img in related_aug_images:
            aug_image_path = os.path.join(augmented_folder, aug_img)

            entry = {
                "id": os.path.splitext(aug_img)[0],  # Remove .png -> Unique ID
                "image": aug_image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                        "<image>\nThis cropped image shows the object position relative to the slot. "
                        "what movement is needed to align it properly?"

                    )                    },
                    {"from": "gpt", "value": combined_instruction}
                ]
            }

            dataset.append(entry)

# Initialize dataset
dataset = []

# Process augmented images
process_augmented_images(fail_mapping_file, fail_image_folder, augmented_folder, dataset)

# Save dataset as JSON
with open(output_json, "w") as json_file:
    json.dump(dataset, json_file, indent=4)

print(f"\nAugmented dataset with 27-class labels saved: {output_json} ")
