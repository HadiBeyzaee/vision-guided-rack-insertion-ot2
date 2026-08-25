"""Grayscale plus CLAHE contrast equalisation.

A preprocessing variant rather than an augmentation: one deterministic output
per input. Local contrast equalisation lifts the slot lip out of shadow in the
back-row deck positions, where reported most of the failures.
"""

import os

# --- Paths (override in your shell or a .env file) ---------------------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
# -----------------------------------------------------------------------
import cv2
import os
from glob import glob

# Input folder with color images
input_dir = os.path.join(BASE_DIR, "data/fail/rgb_cropped2")

# Output folder for processed grayscale CLACHE images
output_dir = os.path.join(BASE_DIR, "data/fail/rgb_cropped2_gray")
os.makedirs(output_dir, exist_ok=True)

# CLAHE setup
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Process all .png, .jpg, .jpeg
image_paths = glob(os.path.join(input_dir, "*.*"))

print(f"[INFO] Found {len(image_paths)} images.")

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"[WARNING] Skipped invalid file: {path}")
        continue

    # Convert to gray + apply CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_clahe = clahe.apply(gray)

    # Save with same filename + _gray suffix
    fname = os.path.basename(path)
    save_path = os.path.join(output_dir, fname.replace(".", "_gray."))

    cv2.imwrite(save_path, gray_clahe)
    print(f"[OK] Saved -> {save_path}")

print("[DONE] All grayscale images saved successfully!")
