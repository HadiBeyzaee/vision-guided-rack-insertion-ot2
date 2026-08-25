"""Sobel edge augmentation: emit an edge-magnitude copy of each image.

Discards colour and texture and keeps gradient magnitude, so the model is
pushed towards the rack and slot boundaries - the features that actually carry
the alignment signal - rather than labware appearance.

Also appears, computed and then discarded, in
complete_system/align_and_insert_cnn.py.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# # Paths
input_folder = os.path.join(BASE_DIR, "slot5_dxyt_1/color_images/camera1_renamed_cropped3")
output_folder = os.path.join(BASE_DIR, "slot5_dxyt_1/color_images/camera1_renamed_cropped3_augmented2")

# (archived path variant removed)
# (archived path variant removed)



os.makedirs(output_folder, exist_ok=True)

# Sobel function
def sobel_edges(gray_img, ksize=3, alpha=1.5):
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(sobel_norm, alpha=alpha, beta=0)

# Number of augmentations (Sobel counts as augmentation here)
num_augmentations = 1

# Process images
for img_file in tqdm(os.listdir(input_folder), desc="Processing Images"):
    if not img_file.lower.endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_folder, img_file)
    image = Image.open(img_path).convert("RGB")

    # Save original
    original_output_path = os.path.join(output_folder, img_file.replace(".png", "_original.png"))
    image.save(original_output_path)

    # Convert to grayscale for Sobel
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Apply Sobel as augmentation
    sobel_img = sobel_edges(gray, ksize=3, alpha=1.5)

    # Save using original code's naming pattern (_aug1.png)
    aug_img_name = img_file.replace(".png", "_aug1.png")
    aug_img_path = os.path.join(output_folder, aug_img_name)
    cv2.imwrite(aug_img_path, sobel_img)

print("\nCompleted! Sobel edge images saved in:", output_folder)
