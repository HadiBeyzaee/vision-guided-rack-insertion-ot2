import os
import json

# ============================================================
# Configuration
# ============================================================

BASE_DIR = "psath/to/your/dataset_directory"

MOVEMENT_TXT = os.path.join(BASE_DIR, "error_data.txt")
IMAGE_MAPPING_JSON = os.path.join(
    BASE_DIR, "path/to/json_file.json"
)
AUGMENTED_IMAGE_DIR = os.path.join(
    BASE_DIR, "path/to/augmented_images"
)

OUTPUT_JSON = os.path.join(
    BASE_DIR, "output_json.json"
)

# ============================================================
# Load movement data
# ============================================================

with open(MOVEMENT_TXT, "r") as f:
    movement_lines = [line.strip for line in f if line.strip]


# ============================================================
# Classification helpers (direction only, no "No Move")
# ============================================================

def classify_dx(dx: float) -> str:
    return "Move Down" if dx >= 0 else "Move Up"


def classify_dy(dy: float) -> str:
    return "Move Right" if dy >= 0 else "Move Left"


def classify_rotation(dtheta: float) -> str:
    return "Rotate Clockwise" if dtheta >= 0 else "Rotate Counterclockwise"


def classify_granularity(dx: float, dy: float, dtheta: float) -> str:
    gx = "Coarse" if abs(dx) > 0.005 else "Fine"
    gy = "Coarse" if abs(dy) > 0.005 else "Fine"
    gr = "Coarse" if abs(dtheta) > 1.0 else "Fine"
    return f"{gx}, {gy}, {gr}"


# ============================================================
# Dataset construction
# ============================================================

def build_dataset(mapping_json: str, augmented_dir: str) -> list:
    with open(mapping_json, "r") as f:
        image_mapping = json.load(f)

    dataset = []

    for _, meta in image_mapping.items:
        base_name = os.path.splitext(meta["new_name"])[0]
        row_idx = meta["row_number"] - 1

        if row_idx >= len(movement_lines):
            continue

        try:
            dx, dy, dtheta = map(float, movement_lines[row_idx].split)
        except ValueError:
            continue

        direction_label = (
            f"{classify_dx(dx)}, "
            f"{classify_dy(dy)}, "
            f"{classify_rotation(dtheta)}"
        )

        granularity_label = classify_granularity(dx, dy, dtheta)

        related_images = [
            img for img in os.listdir(augmented_dir)
            if img.startswith(base_name)
        ]

        for img_name in related_images:
            img_id = os.path.splitext(img_name)[0]
            img_path = os.path.join(augmented_dir, img_name)

            dataset.append({
                "id": img_id,
                "image": img_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\n"
                            "This cropped image shows the object position and orientation "
                            "relative to the slot. What movement (Move Up, Move Down, "
                            "Move Left, Move Right) and what rotation "
                            "(Rotate Clockwise or Rotate Counterclockwise) are needed "
                            "to align it properly?"
                        )
                    },
                    {
                        "from": "gpt",
                        "value": direction_label
                    },
                    {
                        "from": "human",
                        "value": (
                            "For the vertical position, horizontal position, and rotation "
                            "of the object, is the misalignment large (coarse) or small (fine)?"
                        )
                    },
                    {
                        "from": "gpt",
                        "value": granularity_label
                    }
                ]
            })

    return dataset


# ============================================================
# Run
# ============================================================

dataset = build_dataset(
    IMAGE_MAPPING_JSON,
    AUGMENTED_IMAGE_DIR
)

with open(OUTPUT_JSON, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset saved to: {OUTPUT_JSON}")
print(f"Total samples: {len(dataset)}")
