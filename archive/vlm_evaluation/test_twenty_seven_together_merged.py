import os
import subprocess
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from PIL import Image

# Paths
model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-camera1-merged-lora")
fail_folder = os.path.join(BASE_DIR, "opentron_station1_test/color_images/camera1_cropped")
error_data_file = os.path.join(BASE_DIR, "opentron_station1_test/error_data.txt")

# Label definitions
movement_x = ["Move Up", "Move Down", "No Move"]
movement_y = ["Move Left", "Move Right", "No Move"]
rotation = ["Rotate Clockwise", "Rotate clockwise", "Rotate Counterclockwise", "Rotate counterclockwise", "No Rotate"]

combined_labels = [f"{x}, {y}, {r}" for x in movement_x for y in movement_y for r in rotation]
correct_counts = {label: 0 for label in combined_labels}
total_counts = {label: 0 for label in combined_labels}
correct_total = 0
total_samples = 0

# Classification thresholds
def classify_movement(dx, dy):
    move_x = "Move Down" if dx > 0.0005 else "Move Up" if dx < -0.0005 else "No Move"
    move_y = "Move Right" if dy > 0.0005 else "Move Left" if dy < -0.0005 else "No Move"
    return move_x, move_y

def classify_rotation(dtheta):
    if -0.1 <= dtheta <= 0.1:
        return "No Rotate"
    return "Rotate Clockwise" if dtheta > 0.1 else "Rotate Counterclockwise"

# Read ground truth movement data
with open(error_data_file, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

def get_ground_truth(image_filename):
    try:
        row_number = int(os.path.splitext(image_filename)[0]) - 1
        if row_number >= len(movement_lines):
            return None
        dx, dy, dtheta = map(float, movement_lines[row_number].split)
        move_x, move_y = classify_movement(dx, dy)
        rot = classify_rotation(dtheta)
        return f"{move_x}, {move_y}, {rot}"
    except Exception:
        return None

# LLaVA inference using pre-merged model
def run_llava(image_path, query):
    command = [
        "python", RUN_LLAVA,
        "--model-path", model_path,
        "--image-file", image_path,
        "--query", query
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip

# Query
query_combined = (
    "This cropped image shows the object position and orientation relative to the slot. "
    "What movement (Move Up, Move Down, Move Left, Move Right, or No Move) and what rotation "
    "(Rotate Clockwise, Rotate Counterclockwise, or No Rotate) are needed to align it properly?"
)

# Main evaluation loop
def process_images(folder):
    global correct_total, total_samples
    image_files = sorted(
        [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')],
        key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else x
    )

    print(f"\nEvaluating {len(image_files)} images in {folder}...\n")
    for img_file in tqdm(image_files, desc="Processing"):
        image_path = os.path.join(folder, img_file)
        ground_truth = get_ground_truth(img_file)
        if not ground_truth:
            print(f"Skipping {img_file}: No ground truth")
            continue

        output = run_llava(image_path, query_combined)
        predicted = next((label for label in combined_labels if label in output), output)
        correct = predicted == ground_truth

        total_samples += 1
        total_counts[ground_truth] += 1
        if correct:
            correct_total += 1
            correct_counts[ground_truth] += 1

        print('---------------------------------------')
        print(f"Image: {img_file}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Predicted: {predicted} {'' if correct else ''}")
        print('---------------------------------------')

# Run
process_images(fail_folder)

# Plot accuracy
accuracy_per_label = {
    label: (correct_counts[label] / total_counts[label] * 100) if total_counts[label] > 0 else 0
    for label in combined_labels
}

plt.figure(figsize=(14, 6))
plt.bar(combined_labels, [accuracy_per_label[label] for label in combined_labels])
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.ylabel("Accuracy (%)")
plt.title("LLaVA Movement & Rotation Classification Accuracy")
plt.ylim(0, 100)
plt.tight_layout
plt.savefig("together.png")
plt.show

# Final accuracy
final_accuracy = (correct_total / total_samples) * 100 if total_samples > 0 else 0
print(f"\nFinal Accuracy: {final_accuracy:.2f}% ({correct_total}/{total_samples})")
