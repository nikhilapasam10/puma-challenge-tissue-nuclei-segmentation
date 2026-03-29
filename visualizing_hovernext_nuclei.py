#Visualizing nuclei segmentation maps in Google Colab (Track 1 - 3 classes)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from google.colab import drive

# Mount Google Drive (skip if already mounted)
drive.mount('/content/drive', force_remount=True)

# Load the .npy file - change path accordingly
file_path = "/content/drive/MyDrive/training_set_primary_roi_090.npy"
data = np.load(file_path, allow_pickle=True).item()

# Use type_map for segmentation
segmentation = data["type_map"]

# Create a color map automatically
num_labels = int(segmentation.max()) + 1
cmap = plt.get_cmap("tab20", num_labels)  # or 'hsv' if more than 20 labels

# Convert labels to RGB
seg_img = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
label_colors = {}
for label in range(num_labels):
    color = (np.array(cmap(label)[:3]) * 255).astype(np.uint8)
    seg_img[segmentation == label] = color
    label_colors[label] = color / 255  # for legend (normalize to 0-1 for matplotlib)

# Display the segmentation mask
plt.figure(figsize=(8, 8))
plt.imshow(seg_img)
plt.axis('off')

# Create legend
legend_elements = [Patch(facecolor=label_colors[label], edgecolor='black',
                         label=f'Label {label}') for label in label_colors]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
