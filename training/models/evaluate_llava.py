"""Score the fine-tuned VLM on held-out images and plot per-class accuracy.

The VLM counterpart to evaluate_cnn.py, shelling out to run_llava.py per image
and matching the generated text against the 27 labels. Same ground-truth
thresholds, so the two are directly comparable.

QUERY must match the prompt the adapter was trained with.

This measures classification accuracy, not insertion success.
"""

import os
import json
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from PIL import Image

# ======================================================
# User Configuration (EDIT THESE)
# ======================================================

MODEL_PATH = "checkpoints/llava_model_lora"
MODEL_BASE = "llava-hf/llava-v1.5-7b"   # Optional. Remove if your model is merged.

TEST_IMAGE_DIR = "data/test_images"
ERROR_DATA_FILE = "data/error_data.txt"

QUERY = (
    "This cropped image shows the object position and orientation relative to the slot. "
    "What movement and rotation are needed to align it properly?"
)

PLOT_OUTPUT = "llava_per_class_accuracy.png"

# ======================================================
# Class Labels (27 total, must match training)
# ======================================================

movement_x = ["Move Up", "Move Down", "No Move"]
movement_y = ["Move Left", "Move Right", "No Move"]
rotation = ["Rotate Clockwise", "Rotate Counterclockwise", "No Rotate"]

CLASS_LABELS = [
    f"{x}, {y}, {r}"
    for x in movement_x
    for y in movement_y
    for r in rotation
]

# Counters
correct_total = 0
total_samples = 0
correct_per_class = {label: 0 for label in CLASS_LABELS}
total_per_class = {label: 0 for label in CLASS_LABELS}

# Movement thresholds (must match training logic)
DX_THRESHOLD = 0.0005
DY_THRESHOLD = 0.0005
ROT_THRESHOLD = 0.2


# ======================================================
# Ground Truth Label Parsing
# ======================================================

def classify_movement(dx, dy):
    if dx > DX_THRESHOLD: mx = "Move Down"
    elif dx < -DX_THRESHOLD: mx = "Move Up"
    else: mx = "No Move"

    if dy > DY_THRESHOLD: my = "Move Right"
    elif dy < -DY_THRESHOLD: my = "Move Left"
    else: my = "No Move"

    return mx, my

def classify_rotation(dtheta):
    if abs(dtheta) <= ROT_THRESHOLD:
        return "No Rotate"
    return "Rotate Clockwise" if dtheta > 0 else "Rotate Counterclockwise"


with open(ERROR_DATA_FILE, "r") as f:
    gt_lines = [line.strip() for line in f.readlines()]

def get_ground_truth(filename):
    try:
        idx = int(os.path.splitext(filename)[0]) - 1
        dx, dy, dtheta = map(float, gt_lines[idx].split())
        mx, my = classify_movement(dx, dy)
        rot = classify_rotation(dtheta)
        return f"{mx}, {my}, {rot}"
    except:
        return None


# ======================================================
# LLaVA Inference Command Wrapper
# ======================================================

def run_llava(image_path):
    cmd = [
        "python", "-m", "llava.eval.run_llava",
        "--model-path", MODEL_PATH,
        "--model-base", MODEL_BASE,
        "--image-file", image_path,
        "--query", QUERY,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


# ======================================================
# Main Evaluation Loop
# ======================================================

def evaluate():
    global correct_total, total_samples

    image_files = sorted(
        [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(".png")],
        key=lambda x: int(re.findall(r'\d+', x)[0])
    )

    print(f"Found {len(image_files)} test images")

    for img_name in tqdm(image_files):
        gt_label = get_ground_truth(img_name)
        if gt_label is None:
            continue

        img_path = os.path.join(TEST_IMAGE_DIR, img_name)
        output = run_llava(img_path)

        predicted_label = next((lbl for lbl in CLASS_LABELS if lbl in output), "Unknown")

        is_correct = predicted_label == gt_label

        total_samples += 1
        total_per_class[gt_label] += 1
        if is_correct:
            correct_total += 1
            correct_per_class[gt_label] += 1

        print("--------------------------------------")
        print(f"Image:       {img_name}")
        print(f"Ground Truth: {gt_label}")
        print(f"Predicted:    {predicted_label}")
        print(f"Correct:      {is_correct}")
        print("--------------------------------------")

    overall_accuracy = (correct_total / total_samples) * 100 if total_samples > 0 else 0
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({correct_total}/{total_samples})")

    return overall_accuracy


# ======================================================
# Plot Results
# ======================================================

def plot_results():
    per_class_accuracy = {
        label: (correct_per_class[label] / total_per_class[label] * 100)
        if total_per_class[label] > 0 else 0
        for label in CLASS_LABELS
    }

    plt.figure(figsize=(14, 6))
    plt.bar(CLASS_LABELS, per_class_accuracy.values(), color="steelblue")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45, ha="right")
    plt.title("Per-Class Accuracy for Movement and Rotation")
    plt.tight_layout()
    plt.ylim(0, 100)
    plt.savefig(PLOT_OUTPUT)
    plt.show()
    print(f"Saved accuracy plot: {PLOT_OUTPUT}")


# ======================================================
# Run
# ======================================================

if __name__ == "__main__":
    evaluate()
    plot_results()

