"""Score the four-turn dual-prompt adapter on both of its questions.

Asks the direction question and the Coarse/Fine question against the same
checkpoint and reports the two accuracies separately. Its existence confirms
the single dual-prompt adapter was trained and evaluated, not only designed.
"""

import os

# --- Paths (override in your shell or a .env file) ---------------------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_REPO     = os.environ.get("LLAVA_REPO", "/opt/LLaVA")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
RUN_LLAVA      = os.path.join(LLAVA_REPO, "llava/eval/run_llava.py")
# -----------------------------------------------------------------------
import os
import subprocess
import json
from tqdm import tqdm
import re

# === CONFIG ===
model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-camera1-crop2-dual-promt-lora")
model_base = LLAVA_BASE

image_folder =     os.path.join(BASE_DIR, "opentron_station15/color_images/camera1_cropped2")
error_data_file = os.path.join(BASE_DIR, "opentron_station15/error_data.txt")

# === PROMPTS ===
prompt_movement = (
    "This cropped image shows the object position and orientation relative to the slot. "
    "What movement (Move Up, Move Down, Move Left, Move Right, or No Move) and what rotation "
    "(Rotate Clockwise, Rotate Counterclockwise, or No Rotate) are needed to align it properly?"
)

prompt_coarse_fine = (
    "For the horizontal position, vertical position, and rotation, is the misalignment large (coarse) or small (fine)?"
)

# === Read ground truth movement data
with open(error_data_file, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# === Classify movement and rotation
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

def classify_coarse_fine(dx, dy, dtheta):
    offset_dx = "Coarse" if abs(dx) > 0.005 else "Fine"
    offset_dy = "Coarse" if abs(dy) > 0.005 else "Fine"
    offset_theta = "Coarse" if abs(dtheta) > 1.0 else "Fine"
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
