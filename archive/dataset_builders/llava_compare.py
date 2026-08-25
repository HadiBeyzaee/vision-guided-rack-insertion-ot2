import os
import json
import random

# Paths
aligned_dir = os.path.join(BASE_DIR, "opentron_align1/color_images/camera2_cropped_renamed")
misaligned_dir = os.path.join(BASE_DIR, "opentron_align4/color_images/camera2_cropped_renamed")
mapping_file = os.path.join(BASE_DIR, "opentron_align4/color_images/image_row_mapping_camera2_renamed.json")
movement_file = os.path.join(BASE_DIR, "opentron_align4/error_data.txt")
output_json = os.path.join(BASE_DIR, "opentron_align4/llava_trio_no_refimage.json")

# Load error data
with open(movement_file, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Classification
def classify_movement(dx, dy):
    x = "No Move"
    y = "No Move"
    if dx > 0.0008:
        x = "Move Left"
    elif dx < -0.0008:
        x = "Move Right"
    if dy > 0.0008:
        y = "Move Down"
    elif dy < -0.0008:
        y = "Move Up"
    return x, y

def classify_rotation(dtheta):
    if -0.1 <= dtheta <= 0.1:
        return "No Rotate"
    elif dtheta > 0.1:
        return "Rotate Clockwise"
    elif dtheta < -0.1:
        return "Rotate Counterclockwise"

def classify_combined(dx, dy, dtheta):
    mx, my = classify_movement(dx, dy)
    rot = classify_rotation(dtheta)
    return f"{mx}, {my}, {rot}"

# Load file mappings
with open(mapping_file, "r") as f:
    mapping_data = json.load(f)

aligned_imgs = [f for f in os.listdir(aligned_dir) if f.endswith(".png") or f.endswith(".jpg")]
if not aligned_imgs:
    raise RuntimeError("No aligned images found.")

# Final dataset
dataset = []

for uid, data in mapping_data.items:
    original_name = data["original_name"]
    new_name = data["new_name"]
    row = data["row_number"]

    if row > len(movement_lines):
        continue

    dx, dy, dtheta = map(float, movement_lines[row - 1].split)
    label = classify_combined(dx, dy, dtheta)
    misaligned_path = os.path.join(misaligned_dir, new_name)
    random_aligned = random.choice(aligned_imgs)
    aligned_path = os.path.join(aligned_dir, random_aligned)

    # 1. Aligned image: no movement needed
    dataset.append({
        "id": os.path.splitext(random_aligned)[0],
        "image": aligned_path,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nThis is the correct alignment. No movement is needed."
            },
            {
                "from": "gpt",
                "value": "No Move, No Move, No Rotate"
            }
        ]
    })

    # 2. Misaligned image: needs adjustment
    dataset.append({
        "id": uid,
        "image": misaligned_path,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nThis image is misaligned. Adjustment is required."
            },
            {
                "from": "gpt",
                "value": "Acknowledged. Misalignment detected."
            }
        ]
    })

    # 3. Misaligned again: ask for movement
    dataset.append({
        "id": uid + "_ask",
        "image": misaligned_path,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nHow should the black rack move to match correct alignment? Describe translation and rotation needed."
            },
            {
                "from": "gpt",
                "value": label
            }
        ]
    })

# Save
with open(output_json, "w") as jf:
    json.dump(dataset, jf, indent=4)

print(f"\nJSON saved with clean entries: {output_json}")
