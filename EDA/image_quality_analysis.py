import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set dataset path
data_dir = r"C:\Users\USER\Documents\AI\computervision\data\maize"

classes = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

quality_scores = {cls: {"brightness": [], "sharpness": []} for cls in classes}

def compute_brightness(img):
    return np.mean(img)

def compute_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Loop through images and compute metrics
for cls in classes:
    folder = os.path.join(data_dir, cls)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        brightness = compute_brightness(img)
        sharpness = compute_sharpness(img)

        quality_scores[cls]["brightness"].append(brightness)
        quality_scores[cls]["sharpness"].append(sharpness)

# ✅ Plot BRIGHTNESS curve
plt.figure(figsize=(10,5))
for cls in classes:
    plt.plot(sorted(quality_scores[cls]["brightness"]), label=cls)
plt.title("Brightness Curve per Disease Category")
plt.xlabel("Images (sorted by brightness)")
plt.ylabel("Brightness Level")
plt.legend()
plt.grid(True)
plt.show()

# ✅ Plot SHARPNESS curve
plt.figure(figsize=(10,5))
for cls in classes:
    plt.plot(sorted(quality_scores[cls]["sharpness"]), label=cls)
plt.title("Sharpness Curve per Disease Category")
plt.xlabel("Images (sorted by sharpness)")
plt.ylabel("Sharpness Score (Variance of Laplacian)")
plt.legend()
plt.grid(True)
plt.show()

# ✅ Summary Stats
for cls in classes:
    print(f"\n{cls} Summary")
    print(f" Avg Brightness: {np.mean(quality_scores[cls]['brightness']):.2f}")
    print(f" Avg Sharpness : {np.mean(quality_scores[cls]['sharpness']):.2f}")
    print(f" Total Images  : {len(quality_scores[cls]['brightness'])}")
