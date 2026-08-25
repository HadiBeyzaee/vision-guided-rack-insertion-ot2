import os
import json

# # Define paths
# txt_file_path =     os.path.join(BASE_DIR, "opentron6/error_data.txt")
# fail_image_folder = os.path.join(BASE_DIR, "opentron6/color_images/camera2_renamed_cropped")
# fail_mapping_file = os.path.join(BASE_DIR, "opentron6/color_images/image_row_mapping_camera2_renamed.json")
# output_json =       os.path.join(BASE_DIR, "opentron6/together_abs.json")

# Define paths
txt_file_path =     os.path.join(BASE_DIR, "opentron_station32/error_data.txt")
fail_image_folder = os.path.join(BASE_DIR, "opentron_station32/color_images/camera1_renamed_cropped2_")
fail_mapping_file = os.path.join(BASE_DIR, "opentron_station32/color_images/image_row_mapping_camera1_renamed.json")
output_json =       os.path.join(BASE_DIR, "opentron_station32/camera1_crop2_wrong_.json")

# Read movement data from TXT file
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

def classify_movement(dx, dy):
    move_x, move_y = "No Move", "No Move"

    if dx > 0.0005:
        move_x = "Move Down"
    elif dx < -0.0005:
        move_x = "Move Up"

    if dy > 0.0005:
        move_y = "Move Right"
    elif dy < -0.0005:
        move_y = "Move Left"

    return move_x, move_y

def classify_rotation(dtheta):
    if -0.2 <= dtheta <= 0.2:
        return "No Rotate"
    elif dtheta > 0.2:
        return "Rotate Clockwise"
    elif dtheta < -0.2:
        return "Rotate Counterclockwise"


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
                        "what movement and what rotation are needed to align it properly?"

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
