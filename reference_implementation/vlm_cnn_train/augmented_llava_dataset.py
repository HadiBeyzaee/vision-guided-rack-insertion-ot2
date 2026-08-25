import os
import json

"""
=====================================================
LLaVA Dataset Builder - Augmented Images
=====================================================

Required pipeline order (run these scripts first):

1. rename_images.py
   - Creates UUID-based filenames
   - Outputs: image_rename_mapping.json

2. crop_images.py
   - Crops renamed images
   - Outputs: /images/renamed_cropped/

3. image_augmentation.py
   - Produces _orig and _augX versions
   - Outputs: /images/augmented/

Then run THIS script to create:
 augmented_llava_dataset.json
"""

# =====================================================
# User Paths (configure for your dataset)
# =====================================================

BASE_DIR = "/project/data"

ERROR_DATA_TXT = os.path.join(BASE_DIR, "error_data.txt")
RENAME_MAPPING_JSON = os.path.join(BASE_DIR, "image_rename_mapping.json")

AUGMENTED_IMG_DIR = os.path.join(BASE_DIR, "images/augmented")

OUTPUT_JSON = os.path.join(BASE_DIR, "augmented_llava_dataset.json")

DX_THRESHOLD = 0.0005
DY_THRESHOLD = 0.0005
ROT_THRESHOLD = 0.2


# =====================================================
# Movement Classification
# =====================================================

def classify_dx(dx):
    if abs(dx) < DX_THRESHOLD:
        return "No Move"
    return "Move Down" if dx > 0 else "Move Up"


def classify_dy(dy):
    if abs(dy) < DY_THRESHOLD:
        return "No Move"
    return "Move Right" if dy > 0 else "Move Left"


def classify_rotation(dtheta):
    if abs(dtheta) <= ROT_THRESHOLD:
        return "No Rotate"
    return "Rotate Clockwise" if dtheta > 0 else "Rotate Counterclockwise"


# =====================================================
# Dataset Construction
# =====================================================

def generate_augmented_dataset(mapping_json, augmented_dir, error_file):
    dataset = []

    with open(mapping_json, "r") as f:
        rename_map = json.load(f)

    with open(error_file, "r") as f:
        movement_lines = [line.strip for line in f.readlines]

    for uid, info in rename_map.items:
        base_name = os.path.splitext(info["new_filename"])[0]
        row_idx = info["row_index"]

        related_imgs = [
            fn for fn in os.listdir(augmented_dir)
            if fn.startswith(base_name) and fn.lower.endswith(".png")
        ]

        if row_idx > len(movement_lines):
            continue

        try:
            dx, dy, dtheta = map(float, movement_lines[row_idx - 1].split)
        except ValueError:
            continue

        label = (
            f"{classify_dx(dx)}, "
            f"{classify_dy(dy)}, "
            f"{classify_rotation(dtheta)}"
        )

        for fn in related_imgs:
            dataset.append({
                "id": os.path.splitext(fn)[0],
                "image": os.path.join(augmented_dir, fn),
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\nWhat movement and rotation are needed "
                            "to correctly align the object with its slot?"
                        )
                    },
                    {"from": "gpt", "value": label}
                ]
            })

    return dataset


# =====================================================
# Main Execution
# =====================================================

if __name__ == "__main__":
    dataset = generate_augmented_dataset(
        mapping_json=RENAME_MAPPING_JSON,
        augmented_dir=AUGMENTED_IMG_DIR,
        error_file=ERROR_DATA_TXT
    )

    with open(OUTPUT_JSON, "w") as jf:
        json.dump(dataset, jf, indent=4)

    print("Dataset saved:", OUTPUT_JSON)
    print("Total entries:", len(dataset))
