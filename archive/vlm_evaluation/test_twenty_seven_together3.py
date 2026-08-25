import os
import subprocess
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import re  # Ensures correct numerical sorting
from PIL import Image  # New import for resizing

# Define Paths
model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-twenty-seven-black-cropped3-dxyt-nospring-lora")
model_base = LLAVA_BASE

# Image folder (test images)
fail_folder =     os.path.join(BASE_DIR, "black_opentron_test/color_images/camera1_cropped3")
error_data_file = os.path.join(BASE_DIR, "black_opentron_test/error_data.txt")

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

# Function to classify movement (dx, dy)
def classify_movement(dx, dy):
    move_x, move_y = "No Move", "No Move"
    if dx > 0.0008:
        move_x = "Move Down"
    elif dx < -0.0008:
        move_x = "Move Up"
    if dy > 0.0008:
        move_y = "Move Right"
    elif dy < -0.0008:
        move_y = "Move Left"
    return move_x, move_y

# Function to classify rotation (dtheta)
def classify_rotation(dtheta):
    if -0.1 <= dtheta <= 0.1:
        return "No Rotate"
    elif dtheta > 0.1:
        return "Rotate Clockwise"
    elif dtheta < -0.1:
        return "Rotate Counterclockwise"

# Read movement data from `error_data.txt` (ground truth labels)
with open(error_data_file, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Function to get ground truth from `error_data.txt`
def get_ground_truth(image_filename):
    try:
        row_number = int(os.path.splitext(image_filename)[0]) - 1
        if row_number >= len(movement_lines):
            return None
        values = list(map(float, movement_lines[row_number].split))
        dx, dy, dtheta = values[0], values[1], values[2]
        move_x, move_y = classify_movement(dx, dy)
        rotation_label = classify_rotation(dtheta)
        return f"{move_x}, {move_y}, {rotation_label}"
    except (ValueError, IndexError):
        return None

# Function to run inference on LLaVA with resized image
def run_llava(image_path, query):
    resized_image_path = "/tmp/resized_input_image.jpg"
    img = Image.open(image_path).convert("RGB")
    img = img.resize((336, 336))
    img.save(resized_image_path)

    command = [
        "python", RUN_LLAVA,
        "--model-path", model_path,
        "--model-base", model_base,
        "--image-file", resized_image_path,
        "--query", query
    ]
    result = subprocess.run(command, capture_output=True, text=True, universal_newlines=True)
    return result.stdout.strip

# Define single classification query
#query_combined = "The small top component is not perfectly aligned with the silver rack in the center. What movement and rotation are required to bring them into precise alignment?"
query_combined = "This image may show a partial or full misalignment between the object and its holder. What movement (Move Up, Move Down, Move Left, Move Right, No Move) and rotation (Rotate Clockwise, Rotate Counterclockwise, No Rotate) are required to properly align the object with the holder?"

#query_combined = "This image shows a misalignment between the object and its intended slot. How should the object move or rotate to align correctly?"
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
        ground_truth_label = get_ground_truth(img_file)
        if ground_truth_label is None:
            print(f"Skipping {img_file}: No valid ground truth found in error_data.txt")
            continue
        model_output = run_llava(image_path, query_combined)
        predicted_label = next((label for label in combined_labels if label in model_output), "Unknown")
        is_correct = predicted_label == ground_truth_label
        total_samples += 1
        total_counts[ground_truth_label] += 1
        if is_correct:
            correct_total += 1
            correct_counts[ground_truth_label] += 1
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
