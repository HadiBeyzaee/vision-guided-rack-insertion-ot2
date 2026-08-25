"""Photometric augmentation via Albumentations.

A harsher recipe than augment_images.py: random brightness/contrast, gamma,
CLAHE, grayscale, solarize and channel inversion. Aimed at the lighting and
reflection variation of the OT-2 deck rather than at generic jitter.

This is the same pipeline that appears - and is then discarded unused - in
complete_system/align_and_insert_cnn.py, which is how we know it was a training
recipe rather than an inference step.

All transforms are photometric, so rack-slot geometry and the pose label are
unaffected. Analysis only; does not touch the robot.
"""

import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# # Define Paths
input_folder = os.path.join(BASE_DIR, "slot5_dxyt_1/color_images/camera1_renamed_cropped5")
output_folder = os.path.join(BASE_DIR, "slot5_dxyt_1/color_images/camera1_renamed_cropped5_augmented")

# (archived path variant removed)
# (archived path variant removed)


os.makedirs(output_folder, exist_ok=True)

# Albumentations augmentation pipeline
augmentations = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
    A.RandomGamma(gamma_limit=(80, 120), p=0.8),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    A.ToGray(p=0.2),
    A.Solarize(threshold=192, p=0.2),
    A.InvertImg(p=0.2),
    ToTensorV2
])

# Number of augmentations per image
num_augmentations = 1  # Change if you want multiple aug versions

# Process images
for img_file in tqdm(os.listdir(input_folder), desc="Augmenting Images"):
    if not img_file.lower.endswith((".png", ".jpg", ".jpeg")):
        continue

    # Load image with OpenCV and convert to RGB
    img_path = os.path.join(input_folder, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Skipping {img_file} (could not read)")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save original
    original_output_path = os.path.join(output_folder, img_file.replace(".png", "_original.png"))
    cv2.imwrite(original_output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Apply augmentations
    for i in range(1, num_augmentations + 1):
        augmented = augmentations(image=image)
        aug_img = augmented["image"]  # This is a Tensor

        # Convert tensor -> NumPy -> BGR for saving
        aug_img_np = aug_img.permute(1, 2, 0).cpu.numpy
        aug_img_np = (aug_img_np * 255).astype(np.uint8) if aug_img_np.max <= 1 else aug_img_np

        aug_img_path = os.path.join(output_folder, img_file.replace(".png", f"_aug{i}.png"))
        cv2.imwrite(aug_img_path, cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR))

print("\nAugmentation Completed! Augmented images saved in:", output_folder)
