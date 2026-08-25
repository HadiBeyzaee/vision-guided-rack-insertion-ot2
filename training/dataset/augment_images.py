"""Apply label-preserving appearance augmentation.

Step 3 of the dataset pipeline. Colour jitter (p=0.8), random grayscale (p=0.3)
and Gaussian blur (p=0.3), writing <base>_orig and <base>_augN alongside each
other. None of these change the rack-slot geometry, so the pose label is
unaffected.

Salt-and-pepper and speckle noise classes are defined but not wired into the
active pipeline, in this script and in every archived variant of it.
"""

import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# =====================================================
# Fixed paths (your real working setup)
# =====================================================

# Path to input images (renamed + cropped)
INPUT_DIR = "/data/images/renamed_cropped"

# Output directory to store augmented copies
OUTPUT_DIR = "/data/images/augmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_AUG_PER_IMAGE = 1  # Change if needed


# =====================================================
# Custom Noise Transforms (optional usage)
# =====================================================

class AddSaltAndPepperNoise:
    def __init__(self, amount=0.01):
        self.amount = amount

    def __call__(self, img):
        img_np = np.array(img)
        num = int(self.amount * img_np.size / img_np.shape[2])
        coords = [np.random.randint(0, i - 1, num) for i in img_np.shape[:2]]
        img_np[coords[0], coords[1]] = 255  # salt
        coords = [np.random.randint(0, i - 1, num) for i in img_np.shape[:2]]
        img_np[coords[0], coords[1]] = 0    # pepper
        return Image.fromarray(img_np)


class AddSpeckleNoise:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, img):
        np_img = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, self.std, np_img.shape)
        noisy = np.clip(np_img + np_img * noise, 0, 1.0)
        return Image.fromarray((noisy * 255).astype(np.uint8))


# =====================================================
# Augmentation Pipeline
# =====================================================

augment = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.3),
    transforms.ToTensor(),
])


# =====================================================
# Main Execution
# =====================================================

def save_augmented_images():
    files = sorted(f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))

    for img_name in tqdm(files, desc="Augmenting"):
        src_path = os.path.join(INPUT_DIR, img_name)
        img = Image.open(src_path).convert("RGB")

        # Save original copy as version 0
        base, ext = os.path.splitext(img_name)
        original_out = os.path.join(OUTPUT_DIR, f"{base}_orig{ext}")
        img.save(original_out)

        # Create augmented versions
        for i in range(1, NUM_AUG_PER_IMAGE + 1):
            aug_tensor = augment(img)
            aug_img = transforms.ToPILImage()(aug_tensor)
            aug_name = f"{base}_aug{i}{ext}"
            aug_img.save(os.path.join(OUTPUT_DIR, aug_name))

    print("Processing completed:", OUTPUT_DIR)


if __name__ == "__main__":
    save_augmented_images()
