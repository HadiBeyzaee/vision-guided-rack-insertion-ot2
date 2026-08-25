import os
import json

# Define paths
txt_file_path =     os.path.join(BASE_DIR, "opentron_station32/error_data.txt")
fail_image_folder = os.path.join(BASE_DIR, "opentron_station32/color_images/camera1_renamed_cropped2")
fail_mapping_file = os.path.join(BASE_DIR, "opentron_station32/color_images/image_row_mapping_camera1_renamed.json")
output_json =       os.path.join(BASE_DIR, "opentron_station32/camera1_crop2_dual_prompt.json")

# Read movement data
with open(txt_file_path, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Movement classification (directional)
def classify_movement(dx, dy):
    move_x = "Move Down" if dx > 0.0004 else "Move Up" if dx < -0.0004 else "No Move"
    move_y = "Move Right" if dy > 0.0004 else "Move Left" if dy < -0.0004 else "No Move"
    return move_x, move_y

def classify_rotation(dtheta):
    if -0.1 <= dtheta <= 0.1:
        return "No Rotate"
    elif dtheta > 0.1:
        return "Rotate Clockwise"
    else:
        return "Rotate Counterclockwise"

# Coarse/Fine classification
def classify_coarse_fine(dx, dy, dtheta):
    offset_dx = "Coarse" if abs(dx) > 0.005 else "Fine"
    offset_dy = "Coarse" if abs(dy) > 0.005 else "Fine"
    offset_theta = "Coarse" if abs(dtheta) > 1.0 else "Fine"
    return f"{offset_dx}, {offset_dy}, {offset_theta}"

# Process images
def process_images(mapping_file, image_folder, dataset):
    with open(mapping_file, "r") as f:
        image_mapping = json.load(f)

    for unique_id, data in image_mapping.items:
        original_name = data["original_name"]
        new_name = data["new_name"]
        row_number = data["row_number"]

        if row_number > len(movement_lines):
            print(f"Skipping {original_name}: No movement data for row {row_number}")
            continue

        values = list(map(float, movement_lines[row_number - 1].split))
        if len(values) < 3:
            print(f"Skipping {original_name}: Invalid movement format")
            continue

        dx, dy, dtheta = values[0], values[1], values[2]

        movement_x, movement_y = classify_movement(dx, dy)
        rotation = classify_rotation(dtheta)
        movement_instruction = f"{movement_x}, {movement_y}, {rotation}"

        coarse_fine_instruction = classify_coarse_fine(dx, dy, dtheta)

        image_path = os.path.join(image_folder, new_name)

        entry = {
            "id": unique_id,
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        "<image>\nThis cropped image shows the object position and orientation relative to the slot. "
                        "What movement (Move Up, Move Down, Move Left, Move Right, or No Move) and what rotation "
                        "(Rotate Clockwise, Rotate Counterclockwise, or No Rotate) are needed to align it properly?"
                    )
                },
                {"from": "gpt", "value": movement_instruction},
                {
                    "from": "human",
                    "value": (
                        "For the horizontal position, vertical position, and rotation, is the misalignment large (coarse) or small (fine)?"
                    )
                },
                {"from": "gpt", "value": coarse_fine_instruction}
            ]
        }

        dataset.append(entry)

# Run
dataset = []
process_images(fail_mapping_file, fail_image_folder, dataset)

with open(output_json, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"\nMulti-question dataset saved: {output_json} ")
