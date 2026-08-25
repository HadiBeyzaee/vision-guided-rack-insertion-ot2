"""Give every capture a UUID filename and record which label row it belongs to.

Step 1 of the dataset pipeline. Copies raw captures to UUID names and writes
image_rename_mapping.json holding {uuid: {original_filename, new_filename,
row_index}}.

The rename matters because augmentation later produces several files per
capture. Keying on a UUID lets every copy resolve back to the same
error_data.txt row without renumbering anything.
"""

import os
import json
import uuid
import re
import shutil


# =====================================================
# User Paths (configure before running)
# =====================================================
# NOTE: these four constants were commented out in the archived copy, so the
# script raised NameError immediately. Restored as configurable defaults.

BASE_DIR = os.environ.get("BASE_DIR", "/data/project")

LABEL_FILE         = os.path.join(BASE_DIR, "error_data.txt")
IMAGE_DIR_ORIGINAL = os.path.join(BASE_DIR, "images/raw")
IMAGE_DIR_RENAMED  = os.path.join(BASE_DIR, "images/renamed")
MAPPING_JSON_PATH  = os.path.join(BASE_DIR, "image_rename_mapping.json")


# =====================================================
# Helpers
# =====================================================

def numerical_sort(filename):
    """
    Sort filenames containing numbers in human order:
    '1.png', '2.png', '10.png'
    instead of: '1.png', '10.png', '2.png'
    """
    return [int(text) if text.isdigit() else text
            for text in re.split(r"(\d+)", filename)]


# =====================================================
# Main logic
# =====================================================

def main():
    # Create output folder if missing
    os.makedirs(IMAGE_DIR_RENAMED, exist_ok=True)

    # Load movement labels
    with open(LABEL_FILE, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Collect image file list
    image_files = sorted(
        [f for f in os.listdir(IMAGE_DIR_ORIGINAL)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=numerical_sort
    )

    if len(image_files) != len(labels):
        print(f"Warning: {len(image_files)} images vs "
              f"{len(labels)} label entries")

    rename_mapping = {}

    # Rename images and build mapping
    for idx, filename in enumerate(image_files):
        old_path = os.path.join(IMAGE_DIR_ORIGINAL, filename)

        unique_id = uuid.uuid4().hex
        new_filename = f"{unique_id}.png"
        new_path = os.path.join(IMAGE_DIR_RENAMED, new_filename)

        # Copy image under new filename
        shutil.copy2(old_path, new_path)

        rename_mapping[unique_id] = {
            "original_filename": filename,
            "new_filename": new_filename,
            "row_index": idx + 1  # 1-based index matching label file
        }

    # Save rename mapping JSON
    with open(MAPPING_JSON_PATH, "w") as jf:
        json.dump(rename_mapping, jf, indent=4)

    print("Renaming completed.")
    print(f"Renamed images saved to: {IMAGE_DIR_RENAMED}")
    print(f"Rename mapping saved to: {MAPPING_JSON_PATH}")


# =====================================================
# Entry point
# =====================================================

if __name__ == "__main__":
    main()
