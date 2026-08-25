"""
merge_datasets.py

Utility script to merge multiple JSON datasets into a single file.
"""

import json
import os


# ============================================================
# Configuration
# ============================================================

INPUT_JSON_FILES = [
    "/path/to/dataset_1.json",
    "/path/to/dataset_2.json",
    "/path/to/dataset_3.json",
]

OUTPUT_JSON_FILE = "/path/to/merged_dataset.json"


# ============================================================
# Helpers
# ============================================================

def load_json(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# Main merge logic
# ============================================================

def merge_datasets(input_files):
    """Merge multiple JSON datasets into one list."""
    merged_data = []
    file_stats = {}

    for file_path in input_files:
        data = load_json(file_path)

        if not isinstance(data, list):
            raise ValueError(f"{file_path} does not contain a list")

        merged_data.extend(data)
        file_stats[file_path] = len(data)

    return merged_data, file_stats


# ============================================================
# Entry point
# ============================================================

def main():
    merged_data, stats = merge_datasets(INPUT_JSON_FILES)

    print("\nDataset merge summary:")
    for path, count in stats.items():
        print(f" - {os.path.basename(path)}: {count} samples")

    print(f"\nTotal merged samples: {len(merged_data)}")

    save_json(merged_data, OUTPUT_JSON_FILE)
    print(f"\nMerged dataset saved to:\n{OUTPUT_JSON_FILE}")


if __name__ == "__main__":
    main()
