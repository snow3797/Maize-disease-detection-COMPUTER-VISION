import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set dataset path
data_dir =  r"C:\Users\USER\Documents\AI\computervision\data\maize"
classes = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

quality_scores = {cls: {"brightness": [], "sharpness": []} for cls in classes}

def compute_brightness(img):
    return np.mean(img)

def compute_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# Loop through folders
for cls in classes:
    folder = os.path.join(data_dir, cls)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        quality_scores[cls]["brightness"].append(compute_brightness(img))
        quality_scores[cls]["sharpness"].append(compute_sharpness(img))


# ✅ Plot BRIGHTNESS curves
plt.figure(figsize=(10,5))
for cls in classes:
    plt.plot(sorted(quality_scores[cls]["brightness"]), label=cls)
plt.title("Brightness Curve per Disease Category")
plt.xlabel("Images (sorted by brightness)")
plt.ylabel("Brightness Level")
plt.legend()
plt.grid(True)
plt.show()

# ✅ Plot SHARPNESS curves
plt.figure(figsize=(10,5))
for cls in classes:
    plt.plot(sorted(quality_scores[cls]["sharpness"]), label=cls)
plt.title("Sharpness Curve per Disease Category")
plt.xlabel("Images (sorted by sharpness)")
plt.ylabel("Sharpness Score (Variance of Laplacian)")
plt.legend()
plt.grid(True)
plt.show()


# ✅ Summary Stats Table
summary = []
for cls in classes:
    avg_b = np.mean(quality_scores[cls]['brightness'])
    avg_s = np.mean(quality_scores[cls]['sharpness'])
    total = len(quality_scores[cls]['brightness'])
    summary.append([cls, f"{avg_b:.2f}", f"{avg_s:.2f}", total])

fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')
table = ax.table(cellText=summary, colLabels=["Class", "Avg Brightness", "Avg Sharpness", "Total Images"], loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# ✅ Save summary table as PNG
plt.savefig("maize_quality_summary.png", dpi=300, bbox_inches='tight')
print("✅ Summary table saved as maize_quality_summary.png")

# ✅ Print Summary
for row in summary:
    print(row)
