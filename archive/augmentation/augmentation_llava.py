import os
import random
import cv2
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as F

# # Define Paths
# input_folder =  os.path.join(BASE_DIR, "camera2_dxyt4/color_images/camera2_align_renamed_cropped1")
# output_folder = os.path.join(BASE_DIR, "camera2_dxyt4/color_images/camera2_align_renamed_cropped1_augmented2")

input_folder =  os.path.join(BASE_DIR, "camera2_dxyt4/color_images/camera2_renamed_cropped1")
output_folder = os.path.join(BASE_DIR, "camera2_dxyt4/color_images/camera2_renamed_cropped1_augmented2")

os.makedirs(output_folder, exist_ok=True)


class AddSaltAndPepperNoise(object):
    def __init__(self, amount=0.01, salt_vs_pepper=0.5):
        self.amount = amount  # % of image pixels to alter
        self.salt_vs_pepper = salt_vs_pepper

    def __call__(self, img):
        np_img = np.array(img)
        h, w, c = np_img.shape
        num_pixels = int(self.amount * h * w)

        # Salt noise (white dots)
        coords = [np.random.randint(0, i - 1, num_pixels) for i in np_img.shape[:2]]
        np_img[coords[0], coords[1]] = 255

        # Pepper noise (black dots)
        coords = [np.random.randint(0, i - 1, num_pixels) for i in np_img.shape[:2]]
        np_img[coords[0], coords[1]] = 0

        return Image.fromarray(np_img)

# Custom transform to add Speckle Noise
class AddSpeckleNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        np_img = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(self.mean, self.std, np_img.shape)
        noisy = np_img + np_img * noise
        noisy = np.clip(noisy, 0, 1.0)
        noisy_uint8 = (noisy * 255).astype(np.uint8)
        return Image.fromarray(noisy_uint8)

# Helper: Random gamma correction
def random_gamma(img, gamma_range=(0.9, 1.2)):
    gamma = random.uniform(*gamma_range)
    return F.adjust_gamma(img, gamma)

# Helper: Add slight noise (optional)
def add_noise(img, noise_level=5):
    np_img = np.array(img).astype(np.int16)
    noise = np.random.randint(-noise_level, noise_level + 1, np_img.shape, dtype=np.int16)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

augmentations = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.ToTensor,
])

# Number of augmentations per image
num_augmentations = 5  # Change this to increase or decrease the number of augmented versions

# Process images
for img_file in tqdm(os.listdir(input_folder), desc="Augmenting Images"):
    if not img_file.endswith((".png", ".jpg", ".jpeg")):
        continue  # Skip non-image files

    # Load image
    img_path = os.path.join(input_folder, img_file)
    image = Image.open(img_path).convert("RGB")  # Convert to RGB to ensure compatibility

    # Save original to output folder
    original_output_path = os.path.join(output_folder, img_file.replace(".png", "_original.png"))
    image.save(original_output_path)

    # Apply augmentations
    for i in range(1, num_augmentations + 1):
        augmented_image = augmentations(image)  # Tensor output
        aug_img_name = img_file.replace(".png", f"_aug{i}.png")
        aug_img_path = os.path.join(output_folder, aug_img_name)

        # Convert tensor back to PIL before saving
        transforms.ToPILImage(augmented_image).save(aug_img_path)


print("\nAugmentation Completed! Augmented images saved in:", output_folder)
