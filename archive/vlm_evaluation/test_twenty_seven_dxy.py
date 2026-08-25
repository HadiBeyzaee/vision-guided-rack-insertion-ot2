import os
import subprocess
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# Paths

# Define Paths
model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-twenty-seven-black-dxy-lora")
model_base = LLAVA_BASE

# # Image folder (test images)
# fail_folder = os.path.join(BASE_DIR, "opentron_new_test/opentron_new_test1/color_images/camera2_cropped")

# # Error data file for ground truth labels
# error_data_file = os.path.join(BASE_DIR, "opentron_new_test/opentron_new_test1/error_data.txt")

fail_folder = os.path.join(BASE_DIR, "black_opentron_test/color_images/camera1_cropped")

error_data_file = os.path.join(BASE_DIR, "black_opentron_test/error_data.txt")

# Define 27-Class Labels (dx, dy, dtheta-based)
movement_x = [ "Move Down","Move Up", "No Move"]  # dx (Up/Down)
movement_y = ["Move Left", "Move Right", "No Move"]  # dy (Left/Right)

translation_labels = [f"{x}, {y}" for x in movement_x for y in movement_y]

# Accuracy tracking
correct_counts = {label: 0 for label in translation_labels}
total_counts =   {label: 0 for label in translation_labels}
correct_total = 0
total_samples = 0

# Function to classify movement (dx, dy)
def classify_translation(dx, dy):
    move_x, move_y = "No Move", "No Move"

    # dx controls Up/Down
    if dx > 0.0008:
        move_x = "Move Down"
    elif dx < -0.0008:
        move_x = "Move Up"

    # dy controls Left/Right
    if dy > 0.0008:
        move_y = "Move Right"
    elif dy < -0.0008:
        move_y = "Move Left"

    return move_x, move_y  # Always return both movement directions

# Load movement data
with open(error_data_file, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

def get_ground_truth(image_filename):
    try:
        row_number = int(os.path.splitext(image_filename)[0]) - 1
        if row_number >= len(movement_lines):
            return None
        values = list(map(float, movement_lines[row_number].split))
        dx, dy = values[0], values[1]
        move_x, move_y = classify_translation(dx, dy)
        return f"{move_x}, {move_y}"
    except (ValueError, IndexError):
        return None


# Run LLaVA
def run_llava(image_path, query):
    command = [
        "python", RUN_LLAVA,
        "--model-path", model_path,
        "--model-base", model_base,
        "--image-file", image_path,
        "--query", query
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip

# Main evaluation loop
def process_images(folder):
    global correct_total, total_samples
    image_files = sorted(
        [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))],
        key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else x
    )

    print(f"\nTesting {len(image_files)} images in: {folder}")
    for img_file in tqdm(image_files, desc="Evaluating"):
        img_path = os.path.join(folder, img_file)
        gt_label = get_ground_truth(img_file)
        if gt_label is None:
            print(f"Skipping {img_file} (no ground truth)")
            continue

        query =  "This cropped image shows the object position relative to the slot. Based on what you see, what movement (Move Up, Move Down, Move Left, Move Right, or No Move) is needed to align the object properly?"
        output = run_llava(img_path, query)
        pred_label = next((label for label in translation_labels if label in output), "Unknown")

        is_correct = pred_label == gt_label

        total_samples += 1
        total_counts[gt_label] += 1
        if is_correct:
            correct_total += 1
            correct_counts[gt_label] += 1

        print("---------------")
        print(f"{img_file}")
        print(f"GT:  {gt_label}")
        print(f"Predicted: {pred_label} {'' if is_correct else ''}")
        print(f"\033[92mCorrect\033[0m" if is_correct else f"\033[91mIncorrect\033[0m")
        print("---------------")

# Run evaluation
process_images(fail_folder)

# Compute & Plot accuracy
acc_per_label = {
    label: (correct_counts[label] / total_counts[label]) * 100 if total_counts[label] else 0
    for label in translation_labels
}

plt.figure(figsize=(10, 5))
plt.bar(translation_labels, [acc_per_label[l] for l in translation_labels], color="green")
plt.ylabel("Accuracy (%)")
plt.title("LLaVA Accuracy for Translation Only (dx, dy)")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 100)
plt.tight_layout
plt.savefig("seperate_dxy.png")
plt.show

# Final stats
final_acc = (correct_total / total_samples) * 100 if total_samples > 0 else 0
print(f"\nFinal Accuracy: {final_acc:.2f}% ({correct_total}/{total_samples})")
