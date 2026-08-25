import os
import json

# # Define paths
# txt_file_path =     os.path.join(BASE_DIR, "opentron5/error_data.txt")
# fail_image_folder = os.path.join(BASE_DIR, "opentron5/color_images/camera2_renamed_cropped")
# fail_mapping_file = os.path.join(BASE_DIR, "opentron5/color_images/image_row_mapping_camera2_renamed.json")
# output_json =       os.path.join(BASE_DIR, "opentron5/dxy_small_abs.json")


# Define paths
txt_file_path =     os.path.join(BASE_DIR, "black_opentron6/error_data.txt")
fail_image_folder = os.path.join(BASE_DIR, "black_opentron6/color_images/camera1_cropped_renamed")
fail_mapping_file = os.path.join(BASE_DIR, "black_opentron6/color_images/image_row_mapping_camera1_renamed.json")
output_json =       os.path.join(BASE_DIR, "black_opentron6/dxy_only_black.json")

# Read movement data from TXT file
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Function to classify dx (horizontal)
def classify_dx(dx):
    if abs(dx) < 0.0008:
        return "No Move"
    elif dx > 0.0008:
        return "Move Down"
    else:
        return "Move Up"

# Function to classify dy (vertical)
def classify_dy(dy):
    if abs(dy) < 0.0008:
        return "No Move"
    elif dy > 0.0008:
        return "Move Right"
    else:
        return "Move Left"

# Process image annotations
def process_images(mapping_file, image_folder, dataset):
    with open(mapping_file, "r") as f:
        image_mapping = json.load(f)

    for unique_id, data in image_mapping.items:
        original_name = data["original_name"]
        new_name = data["new_name"]
        row_number = data["row_number"]

        image_path = os.path.join(image_folder, new_name)

        if row_number > len(movement_lines):
            print(f"Skipping {original_name}: No movement data for row {row_number}")
            continue

        movement_data = movement_lines[row_number - 1]
        values = list(map(float, movement_data.split))
        if len(values) < 2:
            print(f"Skipping {original_name}: Invalid format in movement data")
            continue

        dx, dy = values[0], values[1]
        dx_label = classify_dx(dx)
        dy_label = classify_dy(dy)
        combined_instruction = f"{dx_label}, {dy_label}"

        entry = {
            "id": unique_id,
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        "<image>\nThis cropped image shows the object position relative to the slot. Based on what you see, what movement (Move Up, Move Down, Move Left, Move Right, or No Move) is needed to align the object properly?"
                    )
                },
                {"from": "gpt", "value": combined_instruction}
            ]
        }

        dataset.append(entry)

# Build and save dataset
dataset = []
process_images(fail_mapping_file, fail_image_folder, dataset)

with open(output_json, "w") as json_file:
    json.dump(dataset, json_file, indent=4)

print(f"\nTranslation-only dataset saved: {output_json} ")
