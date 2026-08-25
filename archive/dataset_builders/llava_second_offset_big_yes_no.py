import os
import json

# Define paths
txt_file_path =     os.path.join(BASE_DIR, "opentron_station32/error_data.txt")
fail_image_folder = os.path.join(BASE_DIR, "opentron_station32/color_images/camera1_renamed_crop3")
fail_mapping_file = os.path.join(BASE_DIR, "opentron_station32/color_images/image_row_mapping_camera1_renamed.json")
output_json =       os.path.join(BASE_DIR, "opentron_station32/camera1_crop3_123_coarse_fine.json")

# Read movement data from TXT file
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Classify if misalignment is coarse or fine
def classify_combined(dx, dy, dtheta):
    offset_dx = "Coarse" if abs(dx) > 0.005 else "Fine"
    offset_dy = "Coarse" if abs(dy) > 0.005 else "Fine"
    offset_dtheta = "Coarse" if abs(dtheta) > 1.0 else "Fine"
    return f"{offset_dx}, {offset_dy}, {offset_dtheta}"

# Function to process images
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
        if len(values) < 3:
            print(f"Skipping {original_name}: Invalid format in movement data")
            continue

        dx, dy, dtheta = values[0], values[1], values[2]

        combined_instruction = classify_combined(dx, dy, dtheta)

        entry = {
            "id": unique_id,
            "image": image_path,
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

# Initialize dataset
dataset = []

# Process fail images
process_images(fail_mapping_file, fail_image_folder, dataset)

# Save dataset as a single JSON file
with open(output_json, "w") as json_file:
    json.dump(dataset, json_file, indent=4)

print(f"\nLLaVA classification dataset saved: {output_json} ")
