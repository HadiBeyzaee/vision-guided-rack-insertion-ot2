import os
import json

"""
==================================================
VLM → CNN Dataset Converter
==================================================

Run this script AFTER:
1. rename_images.py
2. crop_images.py
3. augment_images.py
4. build_llava_dataset.py   (creates augmented_llava_dataset.json)

This script extracts:
- image paths
- classification label

Output format (for CNN training):
[
  {
    "image": "/path/to/img.png",
    "label": "Move Up, Move Left, Rotate Clockwise"
  },
  ...
]
"""

# ==================================================
# User Configuration (edit before running)
# ==================================================

INPUT_JSON = "/data/project/augmented_llava_dataset.json"
OUTPUT_JSON = "/data/project/cnn_training_dataset.json"


# ==================================================
# Conversion Logic
# ==================================================

def build_cnn_dataset(vlm_json, cnn_json_out):
    with open(vlm_json, "r") as f:
        samples = json.load(f)

    cnn_data = []
    missing = 0

    for entry in samples:
        image_path = entry.get("image")
        conversations = entry.get("conversations", [])

        if len(conversations) < 2:
            missing += 1
            continue

        label = conversations[1].get("value", "").strip
        if not image_path or not label:
            missing += 1
            continue

        cnn_data.append({
            "image": image_path,
            "label": label
        })

    with open(cnn_json_out, "w") as f:
        json.dump(cnn_data, f, indent=4)

    print("CNN dataset saved:", cnn_json_out)
    print("Total samples:", len(cnn_data))
    print("Skipped entries (missing fields):", missing)


# ==================================================
# Execute
# ==================================================

if __name__ == "__main__":
    build_cnn_dataset(INPUT_JSON, OUTPUT_JSON)
