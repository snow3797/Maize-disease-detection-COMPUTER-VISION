import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# Set dataset path
data_dir = r"C:\Users\USER\Documents\AI\computervision\data\maize"  # change to your dataset path

# List class folders
classes = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# Count images in each class
class_counts = {}
for cls in classes:
    class_folder = os.path.join(data_dir, cls)
    class_counts[cls] = len(os.listdir(class_folder))

print("Dataset Image Counts:", class_counts)

# ---------- Visualize Class Distribution ----------
plt.figure(figsize=(8,5))
plt.bar(class_counts.keys(), class_counts.values())
plt.title("Class Distribution of Maize Leaf Dataset")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.show()

# ---------- Display Random Images ----------
def show_random_images(data_dir, classes, n=8):
    plt.figure(figsize=(12,8))
    for i in range(n):
        cls = random.choice(classes)
        img_name = random.choice(os.listdir(os.path.join(data_dir, cls)))
        img_path = os.path.join(data_dir, cls, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 4, i+1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_random_images(data_dir, classes)

# ---------- Image Quality & Dimensions ----------
img_sizes = []
for cls in classes:
    folder = os.path.join(data_dir, cls)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img_sizes.append(img.shape)

# Show average size & sample shapes
heights = [s[0] for s in img_sizes]
widths = [s[1] for s in img_sizes]

print(f"Average Image Height: {np.mean(heights):.2f}")
print(f"Average Image Width: {np.mean(widths):.2f}")

# Plot image shapes distribution
plt.figure(figsize=(8,5))
plt.hist(heights, bins=20, alpha=0.6, label='Heights')
plt.hist(widths, bins=20, alpha=0.6, label='Widths')
plt.title("Image Dimension Distribution")
plt.xlabel("Pixels")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# ---------- Check for corrupted images ----------
bad_images = []
for cls in classes:
    folder = Path(data_dir) / cls
    for img_name in os.listdir(folder):
        img_path = folder / img_name
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                bad_images.append(img_path)
        except:
            bad_images.append(img_path)

print(f"Corrupted or unreadable images found: {len(bad_images)}")
if bad_images:
    print("List of corrupted files:", bad_images[:5])
