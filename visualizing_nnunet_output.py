#Code to visualize nnU-Net outputs in Google Colab (upload files to Google Drive and change path)
from google.colab import drive
drive.mount('/content/drive')

# Install required packages if not already installed
!pip install matplotlib numpy pillow

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from PIL import Image

# === CONFIG ===
png_path = "/content/drive/MyDrive/NNUNET_OUTPUT/inference_2d_ensemble_best/training_set_primary_roi_070.png"

# === READ IMAGE ===
image = np.array(Image.open(png_path))
print("Image shape:", image.shape)

# If RGB, take one channel (safety)
if image.ndim == 3:
    image = image[:, :, 0]

image = image.astype(np.int32)

# === FIND UNIQUE LABELS ===
unique_labels = np.unique(image)
num_classes = len(unique_labels)

print("Unique labels:", unique_labels)
print("Number of classes:", num_classes)

# === CREATE COLORMAP (label-safe) ===
colors = plt.cm.get_cmap("tab20", num_classes)
cmap = ListedColormap(colors(range(num_classes)))

# Preserve true label values
bounds = np.append(unique_labels, unique_labels[-1] + 1)
norm = BoundaryNorm(bounds, cmap.N)

# === DISPLAY IMAGE ===
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap=cmap, norm=norm, interpolation="nearest")
plt.axis("off")

# === LEGEND ===
legend_elements = [
    Patch(facecolor=colors(i), label=f"Class {val}")
    for i, val in enumerate(unique_labels)
]

plt.legend(
    handles=legend_elements,
    title="Segmentation Classes",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.show()
