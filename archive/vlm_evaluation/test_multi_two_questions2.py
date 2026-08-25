import os
import subprocess
import json
from tqdm import tqdm
import re

# === CONFIG ===
model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-camera2-crop2-multi-8class-lora")
model_base = LLAVA_BASE

image_folder =     os.path.join(BASE_DIR, "opentron_station1_test/color_images/camera2_cropped2")
error_data_file = os.path.join(BASE_DIR, "opentron_station1_test/error_data.txt")

# === PROMPTS ===
prompt_movement = (
    "This cropped image shows the object position and orientation relative to the slot. "
                        "What movement and rotation are needed to align it properly?"
)

prompt_coarse_fine = (
    "For each of the following: vertical position, horizontal position, and rotation, "
        "is the misalignment large (coarse) or small (fine)?"
)

# === Read ground truth movement data
with open(error_data_file, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Movement classification (no "No Move")
def classify_movement(dx, dy):
    move_x = "Move Down" if dx >= 0 else "Move Up"
    move_y = "Move Right" if dy >= 0 else "Move Left"
    return move_x, move_y

# Rotation classification (no "No Rotate")
def classify_rotation(dtheta):
    return "Rotate Clockwise" if dtheta > 0 else "Rotate Counterclockwise"


def classify_coarse_fine(dx, dy, dtheta):
    offset_dx = "Coarse" if abs(dx) > 0.005 else "Fine"
    offset_dy = "Coarse" if abs(dy) > 0.005 else "Fine"
    offset_theta = "Coarse" if abs(dtheta) > 0.8 else "Fine"
    return f"{offset_dx}, {offset_dy}, {offset_theta}"

# === Run LLaVA on an image with a query
def run_llava(image_path, query):
    command = [
        "python", RUN_LLAVA,
        "--model-path", model_path,
        "--model-base", model_base,
        "--image-file", image_path,
        "--query", query
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    lines = result.stdout.strip.split("\n")

    # Only return the final model response (ignore logs)
    for line in reversed(lines):
        if line.strip and not line.strip.startswith("["):
            return line.strip

    return "Unknown"


# === Evaluate all images
total = 0
correct_move = 0
correct_fine = 0

image_files = sorted(
    [f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")],
    key=lambda x: int(re.findall(r'\d+', x)[0])
)

print(f"\nTesting on {len(image_files)} images...\n")

for img_file in tqdm(image_files):
    img_path = os.path.join(image_folder, img_file)
    row = int(os.path.splitext(img_file)[0]) - 1
    if row >= len(movement_lines):
        continue
    dx, dy, dtheta = map(float, movement_lines[row].split)

    gt_move_x, gt_move_y = classify_movement(dx, dy)
    gt_rotation = classify_rotation(dtheta)
    gt_movement = f"{gt_move_x}, {gt_move_y}, {gt_rotation}"
    gt_fine = classify_coarse_fine(dx, dy, dtheta)

    pred_movement = run_llava(img_path, prompt_movement)
    pred_movement_label = next((lbl for lbl in [gt_movement] if lbl in pred_movement), pred_movement)

    pred_fine = run_llava(img_path, prompt_coarse_fine)
    pred_fine_label = next((lbl for lbl in [gt_fine] if lbl in pred_fine), pred_fine)

    is_move_correct = pred_movement_label.strip == gt_movement
    is_fine_correct = pred_fine_label.strip == gt_fine

    total += 1
    correct_move += int(is_move_correct)
    correct_fine += int(is_fine_correct)

    print(f"\n{img_file}")
    print(f"GT-Move:   {gt_movement}")
    print(f"PRED-Move: {pred_movement_label} {'' if is_move_correct else ''}")
    print(f"GT-Fine:   {gt_fine}")
    print(f"PRED-Fine: {pred_fine_label} {'' if is_fine_correct else ''}")
    print("------------------------------------------------")

# === Final Accuracy
acc_move = correct_move / total * 100 if total else 0
acc_fine = correct_fine / total * 100 if total else 0

print(f"\nFinal Movement Accuracy: {acc_move:.2f}% ({correct_move}/{total})")
print(f"Final Coarse/Fine Accuracy: {acc_fine:.2f}% ({correct_fine}/{total})")
