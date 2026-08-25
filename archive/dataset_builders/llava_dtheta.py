import os
import json

# # Define paths
# txt_file_path =     os.path.join(BASE_DIR, "opentron1/error_data.txt")
# fail_image_folder = os.path.join(BASE_DIR, "opentron1/color_images/camera2_renamed_cropped")
# fail_mapping_file = os.path.join(BASE_DIR, "opentron1/color_images/image_row_mapping_camera2_renamed.json")
# output_json =       os.path.join(BASE_DIR, "opentron1/dtheta_small_abs.json")

# Define paths
txt_file_path =     os.path.join(BASE_DIR, "black_opentron1/error_data.txt")
fail_image_folder = os.path.join(BASE_DIR, "black_opentron1/color_images/camera1_renamed_cropped")
fail_mapping_file = os.path.join(BASE_DIR, "black_opentron1/color_images/image_row_mapping_camera1_renamed.json")
output_json =       os.path.join(BASE_DIR, "black_opentron1/theta_only_black.json")

# Read rotation data from TXT file
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

def classify_rotation(dtheta):
    if -0.1 <= dtheta <= 0.1:
        return "No Rotate"
    elif dtheta > 0.1:
        return "Rotate Clockwise"
    elif dtheta < -0.1:
        return "Rotate Counterclockwise"


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

        dtheta = values[2]

        # Only classify rotation
        rotation_instruction = classify_rotation(dtheta)

        # Structured LLaVA prompt with only rotation
        entry = {
            "id": unique_id,
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        "<image>\nBased on what you see, what rotation (Rotate Clockwise, Rotate Counterclockwise, or No Rotate) is needed to align the object properly?"
                    )
                },
                {"from": "gpt", "value": rotation_instruction}
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

print(f"\nRotation-only dataset saved: {output_json} ")
