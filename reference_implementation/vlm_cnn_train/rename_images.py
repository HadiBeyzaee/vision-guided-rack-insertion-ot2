import os
import json
import uuid
import re
import shutil


# LABEL_FILE = "/data/project/error_data.txt"
# IMAGE_DIR_ORIGINAL = "/data/project/images/raw"
# IMAGE_DIR_RENAMED = "/data/project/images/renamed"
# MAPPING_JSON_PATH = "/data/project/image_rename_mapping.json"


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
