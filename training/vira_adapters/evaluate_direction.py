"""Score a ViRA direction adapter on held-out images.

Shells out to run_llava.py per image and matches the generated text against the
label list. The query string must match the prompt the adapter was trained with
and the one its server sends, or accuracy collapses at deployment while looking
fine here.
"""

import os

# --- Paths (override in your shell or a .env file) ---------------------
# BASE_DIR       : dataset root holding error_data.txt and color_images/
# LLAVA_REPO     : checkout of haotian-liu/LLaVA (provides llava/eval/run_llava.py)
# LLAVA_BASE     : base llava-v1.5-7b weights
# CHECKPOINT_DIR : directory holding the fine-tuned LoRA adapters
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_REPO     = os.environ.get("LLAVA_REPO", "/opt/LLaVA")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
RUN_LLAVA      = os.path.join(LLAVA_REPO, "llava/eval/run_llava.py")
# -----------------------------------------------------------------------
import os
import subprocess
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import re  # Ensures correct numerical sorting
from PIL import Image  # New import for resizing

# Define Paths
model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-wrong-crop2-camera1-lora")
model_base = LLAVA_BASE

# Image folder (test images)
fail_folder =     os.path.join(BASE_DIR, "opentron_wrong_test/opentron_wrong_test2/color_images/camera1_cropped2")
error_data_file = os.path.join(BASE_DIR, "opentron_wrong_test/opentron_wrong_test2/error_data.txt")

# Define 27-Class Labels (dx, dy, dtheta-based)
movement_x = ["Move Up", "Move Down", "No Move"]  # dx (Up/Down)
movement_y = ["Move Left", "Move Right", "No Move"]  # dy (Left/Right)
rotation = ["Rotate Clockwise", "Rotate clockwise", "Rotate Counterclockwise", "Rotate counterclockwise", "No Rotate"]

# Ensure the same order is used everywhere
combined_labels = [
    f"{x}, {y}, {r}"
    for x in movement_x
    for y in movement_y
    for r in rotation
]

# Accuracy tracking per label
correct_counts = {label: 0 for label in combined_labels}
total_counts = {label: 0 for label in combined_labels}

# Accuracy counters
correct_total = 0
total_samples = 0

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

# Read movement data from `error_data.txt` (ground truth labels)
with open(error_data_file, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Function to get ground truth from `error_data.txt`
def get_ground_truth(image_filename):
    try:
        row_number = int(os.path.splitext(image_filename)[0]) - 1  # Convert filename (e.g., "1.png") -> row index
        if row_number >= len(movement_lines):
            return None  # No data for this row

        values = list(map(float, movement_lines[row_number].split))
        dx, dy, dtheta = values[0], values[1], values[2]

        move_x, move_y = classify_movement(dx, dy)
        rotation_label = classify_rotation(dtheta)

        # Fix the order to match `combined_labels`
        return f"{move_x}, {move_y}, {rotation_label}"
    except (ValueError, IndexError):
        return None  # Skip invalid rows


# Function to run inference on LLaVA
def run_llava(image_path, query):
    command = [
        "python", RUN_LLAVA,
        "--model-path", model_path,
        "--model-base", model_base,
        "--image-file", image_path,
        "--query", query
    ]
    result = subprocess.run(command, capture_output=True, text=True, universal_newlines=True)
    return result.stdout.strip

# Define single classification query
#query_combined =  "The small top component is not perfectly aligned with the silver rack in the center. What movement and rotation are required to bring them into precise alignment?"
#query_combined =  "The object is slightly misaligned with the silver slot in the center. What movement and rotation are needed to align them?"

#query_combined = "This image may show a partial or full misalignment between the object and its holder. What movement (Move Up, Move Down, Move Left, Move Right, No Move) and rotation (Rotate Clockwise, Rotate Counterclockwise, or No Rotate) are required to properly align the object with the holder?"

query_combined = "This cropped image shows the object position and orientation relative to the slot. what movement and what rotation are needed to align it properly?"

def process_images(folder):
    global correct_total, total_samples

    image_files = sorted(
        [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')],
        key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else x
    )

    print(f"\nEvaluating images in {folder} ({len(image_files)} images)...")

    for img_file in tqdm(image_files, desc=f"Processing {folder}"):
        image_path = os.path.join(folder, img_file)

        # Get ground truth label
        ground_truth_label = get_ground_truth(img_file)
        if ground_truth_label is None:
            print(f"Skipping {img_file}: No valid ground truth found in error_data.txt")
            continue

        # Run LLaVA inference
        model_output = run_llava(image_path, query_combined)

        # Extract predicted label from LLaVA output
        predicted_label = next((label for label in combined_labels if label in model_output), model_output)


        # Determine correctness
        is_correct = predicted_label == ground_truth_label

        # Update accuracy counters
        total_samples += 1
        total_counts[ground_truth_label] += 1
        if is_correct:
            correct_total += 1
            correct_counts[ground_truth_label] += 1

        # Print summary
        print('---------------------------------------')
        print(f"Image file: {img_file}")
        print(f"Ground Truth: {ground_truth_label}")
        print(f"Predicted: {predicted_label} {'' if is_correct else ''}")
        print(f"\033[92mCorrect\033[0m" if is_correct else f"\033[91mIncorrect\033[0m")
        print('---------------------------------------')

# Process images
process_images(fail_folder)

# Compute accuracy per label
accuracy_per_label = {
    label: (correct_counts[label] / total_counts[label] * 100) if total_counts[label] > 0 else 0 for label in combined_labels
}

# Plot accuracy with combined labels
plt.figure(figsize=(12, 6))
plt.bar(combined_labels, [accuracy_per_label[label] for label in combined_labels], color="blue")
plt.ylabel("Accuracy (%)")
plt.title("LLaVA Accuracy for Movement & Rotation")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.ylim(0, 100)
plt.savefig("together.png")
plt.show

# Print final accuracy
final_accuracy = (correct_total / total_samples) * 100 if total_samples > 0 else 0
print(f"\n**Final Accuracy: {final_accuracy:.2f}% ({correct_total}/{total_samples})**")


