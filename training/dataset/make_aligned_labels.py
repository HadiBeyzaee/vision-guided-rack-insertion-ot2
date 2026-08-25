"""Write a zero label row per image, for the no-correction subset.

The models learn when to stop from images that need no correction: 1500 of them
in the ViRA set, 2000 in the single-stage set. Those were captured with the rack
already seated rather than by commanding a small offset, so they have no
error_data.txt of their own. This writes one "0.0 0.0 0.0" row per image.

Note the labels are exactly zero, while described this subset as
carrying small nonzero offsets up to 0.5 mm and 0.1 deg. The real residual is
whatever the physical seating left behind; it is not measured.

Run renumber_images.py first. Analysis only; does not touch the robot.
"""

import os

# --- Paths (override in your shell or a .env file) ---------------------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
# -----------------------------------------------------------------------
import os

folder_path = os.path.join(BASE_DIR, "camera2_dxyt4/color_images/camera2_align")
output_path = os.path.join(BASE_DIR, "camera2_dxyt4/error_data_align.txt")


# Count only .png files (case-insensitive)
image_count = len([f for f in os.listdir(folder_path) if f.lower.endswith('.png')])

print(f"Total PNG images in folder: {image_count}")

with open(output_path, "w") as f:
    for _ in range(image_count):
        f.write("0.0 0.0 0.0\n")

print(f"Created file with 400 rows: {output_path}")
