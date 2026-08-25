import os
import json

# Define paths
txt_file_path =     os.path.join(BASE_DIR, "opentron_stiff1/error_data.txt")
fail_image_folder = os.path.join(BASE_DIR, "opentron_stiff1/color_images/camera2_cropped_renamed")
fail_mapping_file = os.path.join(BASE_DIR, "opentron_stiff1/color_images/image_row_mapping_camera2_renamed.json")
output_json =       os.path.join(BASE_DIR, "opentron_stiff1/offset_amount.json")

# Thresholds
DX_THRESHOLD = 0.007
DY_THRESHOLD = 0.007
DTHETA_THRESHOLD = 1.0

# Read movement data
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Generate dataset
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

        values = list(map(float, movement_lines[row_number - 1].split))
        if len(values) < 3:
            print(f"Skipping {original_name}: Invalid format in movement data")
            continue

        dx, dy, dtheta = values[0], values[1], values[2]

        # Compute yes/no flags in dx, dy, dtheta order
        dx_flag     = "yes" if abs(dx) > DX_THRESHOLD else "no"
        dy_flag     = "yes" if abs(dy) > DY_THRESHOLD else "no"
        dtheta_flag = "yes" if abs(dtheta) > DTHETA_THRESHOLD else "no"

        gpt_reply = f"{dx_flag} {dy_flag} {dtheta_flag}"

        conversations = [
            {
                "from": "human",
                "value": "<image>\nIs the offset between the silver rack holder and the black rack large in any of these aspects: 1) Vertical alignment, 2) Horizontal alignment, 3) Rotation? Answer with: yes/no yes/no yes/no"
            },
            {
                "from": "gpt",
                "value": gpt_reply
            }
        ]

        dataset.append({
            "id": unique_id,
            "image": image_path,
            "conversations": conversations
        })

# Initialize and run
dataset = []
process_images(fail_mapping_file, fail_image_folder, dataset)

# Save
with open(output_json, "w") as json_file:
    json.dump(dataset, json_file, indent=4)

print(f"\nCorrection dataset saved with dx -> dy -> dtheta order: {output_json} ")
