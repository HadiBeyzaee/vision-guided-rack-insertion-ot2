import os
import subprocess
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# Define Paths
model_path = os.path.join(CHECKPOINT_DIR, "llava-v1.5-7b-opentron-offset-amount-lora")
model_base = LLAVA_BASE
fail_folder = os.path.join(BASE_DIR, "opentron_new_test/opentron_new_test1/color_images/camera2_cropped")
error_data_file = os.path.join(BASE_DIR, "opentron_new_test/opentron_new_test1/error_data.txt")

# Thresholds for dx, dy, dtheta
DX_THRESHOLD = 0.007
DY_THRESHOLD = 0.007
DTHETA_THRESHOLD = 1.0

# Read ground truth offsets
with open(error_data_file, "r") as f:
    movement_lines = [line.strip for line in f.readlines]

# Get ground truth label in format: "yes no yes"
def get_ground_truth(image_filename):
    try:
        row_number = int(os.path.splitext(image_filename)[0]) - 1
        if row_number >= len(movement_lines):
            return None

        values = list(map(float, movement_lines[row_number].split))
        dx, dy, dtheta = values[0], values[1], values[2]

        dx_flag = "yes" if abs(dx) > DX_THRESHOLD else "no"
        dy_flag = "yes" if abs(dy) > DY_THRESHOLD else "no"
        dtheta_flag = "yes" if abs(dtheta) > DTHETA_THRESHOLD else "no"

        return f"{dx_flag} {dy_flag} {dtheta_flag}"
    except Exception as e:
        return None

# Run LLaVA model on image
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

# Custom yes/no query
query_yesno = (
    "Is the offset between the silver rack holder and the black rack large in any of these aspects: 1) Vertical alignment, 2) Horizontal alignment, 3) Rotation? Answer with: yes/no yes/no yes/no"
)

# Evaluation counters
correct_total = 0
total_samples = 0
label_match_counts = {}

# Main evaluation loop
def process_images(folder):
    global correct_total, total_samples

    image_files = sorted(
        [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')],
        key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else x
    )

    print(f"\nEvaluating {len(image_files)} test images...")

    for img_file in tqdm(image_files, desc="Testing"):
        image_path = os.path.join(folder, img_file)

        # Ground truth
        ground_truth = get_ground_truth(img_file)
        if ground_truth is None:
            print(f"Skipping {img_file}: No valid ground truth.")
            continue

        # Run model
        model_output = run_llava(image_path, query_yesno)

        # Normalize and match format
        prediction = None
        for line in model_output.splitlines:
            parts = line.lower.strip.split
            if len(parts) == 3 and all(p in ["yes", "no"] for p in parts):
                prediction = " ".join(parts)
                break

        if prediction is None:
            print(f"Skipping {img_file}: Invalid model output")
            continue

        # Count stats
        is_correct = prediction == ground_truth
        total_samples += 1
        if is_correct:
            correct_total += 1

        label_match_counts[ground_truth] = label_match_counts.get(ground_truth, {"correct": 0, "total": 0})
        label_match_counts[ground_truth]["total"] += 1
        if is_correct:
            label_match_counts[ground_truth]["correct"] += 1

        # Print result
        print('---------------------------------------')
        print(f" Image: {img_file}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Predicted:    {prediction} {'' if is_correct else ''}")
        print('---------------------------------------')

# Run evaluation
process_images(fail_folder)

# Final accuracy
final_accuracy = (correct_total / total_samples) * 100 if total_samples > 0 else 0
print(f"\nFinal Accuracy: {final_accuracy:.2f}% ({correct_total}/{total_samples})")

# Optional: Accuracy by label
print("\nAccuracy by Label:")
for label, stats in label_match_counts.items:
    acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    print(f"  {label}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
