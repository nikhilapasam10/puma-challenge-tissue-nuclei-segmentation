#Visualizing png segmentation maps through Google Colab

from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install matplotlib numpy pillow

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

# path to the png segmentation map to visualize
png_path = "/content/drive/MyDrive/Dataset998_Melanom/labelsTr/training_set_primary_roi_070.png"

# === READ PNG LABEL MASK ===
image = np.array(Image.open(png_path))

print("Original image shape:", image.shape)

# If PNG is RGB, take one channel (label masks usually have identical channels)
if image.ndim == 3:
    image = image[:, :, 0]

# Ensure integer labels (PNG may load as uint8 already)
image = image.astype(np.int32)

print("Processed image shape:", image.shape)

# === FIND UNIQUE LABELS ===
unique_vals = np.unique(image)
num_unique = len(unique_vals)

print("Unique label values:", unique_vals)
print("Number of different labels:", num_unique)

# === CREATE DISCRETE COLORMAP ===
colors = plt.cm.get_cmap('tab20', num_unique)
cmap = ListedColormap(colors(np.arange(num_unique)))

# === DISPLAY IMAGE ===
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap=cmap, interpolation='nearest')
plt.axis('off')

# === CREATE LEGEND ===
legend_elements = [
    Patch(facecolor=colors(i), label=str(val))
    for i, val in enumerate(unique_vals)
]

plt.legend(
    handles=legend_elements,
    title="Label Values",
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.show()
