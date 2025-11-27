import os
import cv2
import random
from glob import glob
import numpy as np

# ---- CONFIG ----
DATASET_DIR = "Dataset"   # your dataset folder
AUG_PER_IMAGE = 4         # how many transformations per image

# Augmentation functions
def random_flip(img):
    return cv2.flip(img, 1) if random.random() < 0.5 else img

def random_rotation(img):
    angle = random.uniform(-90, 90)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_brightness(img):
    factor = random.uniform(0.6, 1.4)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_blur(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

# List of transforms to apply randomly
TRANSFORMS = [random_flip, random_rotation, random_brightness, random_blur]


# ---- MAIN SCRIPT ----
i = 0
for class_dir in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_dir)
    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {class_dir}")

    # Get existing images
    images = sorted(glob(os.path.join(class_path, "*.*")),
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # Determine starting index for new augmented images
    if images:
        last_index = int(os.path.splitext(os.path.basename(images[-1]))[0])
    else:
        last_index = 0

    # Process each image
    for img_path in images:
        img = cv2.imread(img_path)

        for i in range(AUG_PER_IMAGE):
            aug_img = img.copy()

            # Apply random transformations
            for transform in TRANSFORMS:
                if random.random() < 0.8:  # 80% chance for each transform
                    aug_img = transform(aug_img)

            # Save with new index
            last_index += 1
            save_path = os.path.join(class_path, f"{last_index}.jpg")
            cv2.imwrite(save_path, aug_img)

    print(f"Finished class {class_dir}. Last image index = {last_index}")

print("✔ Augmentation completed successfully!")
