"""Copy a folder of captures to 1.png ... N.png in sorted order.

Pairs with make_aligned_labels.py: the aligned captures arrive with arbitrary
names, and the dataset builders index images by number against a label file.

Copies rather than moves, so the originals survive a mistake.
"""

import os

# --- Paths (override in your shell or a .env file) ---------------------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
# -----------------------------------------------------------------------
import os
import shutil

# Source and destination folders
source_folder =      os.path.join(BASE_DIR, "camera2_dxyt4/color_images/camera2_align")
destination_folder = os.path.join(BASE_DIR, "camera2_dxyt4/color_images/camera2_align_new")

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Get sorted list of .png files
image_files = sorted([f for f in os.listdir(source_folder) if f.lower.endswith('.png')])

# Copy and rename
for i, filename in enumerate(image_files, start=1):
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(destination_folder, f"{i}.png")
    shutil.copy(src_path, dst_path)

print(f"Copied and renamed {len(image_files)} images to {destination_folder}")
