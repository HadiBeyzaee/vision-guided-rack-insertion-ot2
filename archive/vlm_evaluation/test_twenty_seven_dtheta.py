import os
import subprocess
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import re  # Ensures correct numerical sorting


model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-twenty-seven-black-theta-lora")
model_base = LLAVA_BASE

# # Image folder (test images)
# fail_folder = os.path.join(BASE_DIR, "opentron_new_test/opentron_new_test1/color_images/camera2_cropped")
# error_data_file = os.path.join(BASE_DIR, "opentron_new_test/opentron_new_test1/corrected_error_data.txt")

fail_folder = os.path.join(BASE_DIR, "opentron_middle_test/color_images/camera1_cropped")
error_data_file = os.path.join(BASE_DIR, "opentron_middle_test/error_data.txt")

rotation_labels = ["Rotate Clockwise", "Rotate clockwise", "Rotate Counterclockwise", "Rotate counterclockwise", "No Rotate"]

# Accuracy tracking per rotation label
correct_counts = {label: 0 for label in rotation_labels}
total_counts = {label: 0 for label in rotation_labels}

# Accuracy counters
correct_total = 0
total_samples = 0

# Function to classify rotation (dtheta)
def classify_rotation(dtheta):
    if -0.1 <= dtheta <= 0.1:
        return "No Rotate"
    elif dtheta > 0.1:
        return "Rotate Clockwise"
    elif dtheta < -0.1:
        return "Rotate Counterclockwise"

# Read movement data from `error_data.txt`
with open(error_data_file, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Function to get ground truth from error_data.txt (ONLY rotation)
def get_ground_truth(image_filename):
    try:
        row_number = int(os.path.splitext(image_filename)[0]) - 1
        if row_number >= len(movement_lines):
            return None

        values = list(map(float, movement_lines[row_number].split))
        dtheta = values[2]  # Only take dtheta (ignore dx, dy)
        return classify_rotation(dtheta)
    except (ValueError, IndexError):
        return None

# Function to run inference on LLaVA (ONLY rotation)
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

# Function to process images
def process_images(folder):
    global correct_total, total_samples
    image_files = sorted(
        [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')],
        key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else x
    )

    print(f"\nEvaluating images in {folder} ({len(image_files)} images)...")

    for img_file in tqdm(image_files, desc=f"Processing {folder}"):
        image_path = os.path.join(folder, img_file)
        ground_truth_rotation = get_ground_truth(img_file)
        if ground_truth_rotation is None:
            print(f"Skipping {img_file}: No valid ground truth found in error_data.txt")
            continue

        # Run LLaVA inference for rotation
        query_rotation = """How should the black rack (bottom) rotate to align with the silver holder above? Choose from: Rotate Clockwise, Rotate Counterclockwise, or No Rotate."""

        model_output_rotation = run_llava(image_path, query_rotation)

        # Extract predicted rotation label
        predicted_rotation = next((label for label in rotation_labels if label in model_output_rotation), "Unknown")

        # Determine correctness
        is_correct = predicted_rotation == ground_truth_rotation

        # Update accuracy counters
        total_samples += 1
        total_counts[ground_truth_rotation] += 1
        if is_correct:
            correct_total += 1
            correct_counts[ground_truth_rotation] += 1

        # Print summary
        print('---------------------------------------')
        print(f"Image file: {img_file}")
        print(f"Ground Truth Rotation: {ground_truth_rotation}")
        print(f"Predicted Rotation: {predicted_rotation}")
        print(f"\033[92mCorrect\033[0m" if is_correct else f"\033[91mIncorrect\033[0m")
        print('---------------------------------------')

# Process images (ONLY rotation, NO translation)
process_images(fail_folder)

# Compute accuracy per label
accuracy_per_label = {
    label: (correct_counts[label] / total_counts[label] * 100) if total_counts[label] > 0 else 0 for label in rotation_labels
}

# Plot accuracy for rotation labels
plt.figure(figsize=(8, 5))
plt.bar(rotation_labels, [accuracy_per_label[label] for label in rotation_labels], color="blue")
plt.ylabel("Accuracy (%)")
plt.title("LLaVA Accuracy for Rotation (dtheta Only)")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 100)
plt.savefig("rotation.png")
plt.show
# Final stats
final_acc = (correct_total / total_samples) * 100 if total_samples > 0 else 0
print(f"\nFinal Accuracy: {final_acc:.2f}% ({correct_total}/{total_samples})")
