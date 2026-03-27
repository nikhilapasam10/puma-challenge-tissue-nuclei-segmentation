'''
This code does the following:

Loads a pathology image.
Loads the pretrained TIA semantic segmentation model.
Splits the image into tiles and predicts tissue types per tile.
Saves raw prediction probabilities.
Processes predictions into class labels.
Visualizes both the raw probability maps and the final segmented image.
'''

# Import libraries
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.utils.misc import imread
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Optional: set figure DPI and background color
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["figure.facecolor"] = "white"

# Path to your input image (update to the correct SOMAI path)
img_file_name = "/data/npasam/PUMA_Dataset/01_training_dataset_tif_ROIs/training_set_primary_roi_090.tif"

# Directory to save predictions
save_dir = "/data/npasam/tia_predictions_sample"

# Load the pretrained segmentor
bcc_segmentor = SemanticSegmentor(
    pretrained_model="fcn_resnet50_unet-bcss",  
    num_loader_workers=4,
    batch_size=4
)

# Run tile-based prediction
output = bcc_segmentor.predict(
    [img_file_name],
    save_dir=save_dir,  # save predictions here
    mode="tile",
    resolution=1.0,
    units="baseline",
    patch_input_shape=[1024, 1024],
    patch_output_shape=[512, 512],
    stride_shape=[512, 512],
    crash_on_exception=True,
)

print("Prediction method output is:", output)

# Load raw prediction
tile_prediction_raw = np.load(output[0][1] + ".raw.0.npy")
print("Raw prediction dimensions:", tile_prediction_raw.shape)

# Convert raw prediction to class labels
tile_prediction = np.argmax(tile_prediction_raw, axis=-1)
print("Processed prediction dimensions:", tile_prediction.shape)

# Load the original image
tile = imread(img_file_name)
print("Input image dimensions:", tile.shape)

# Show prediction probability maps for each class
fig = plt.figure(figsize=(15, 5))
label_names_dict = {
    0: "Tumour",
    1: "Stroma",
    2: "Inflammatory",
    3: "Necrosis",
    4: "Others",
}
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    plt.imshow(tile_prediction_raw[:, :, i])
    plt.xlabel(label_names_dict[i])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
fig.suptitle("Raw prediction maps for different classes", y=0.65)

# Show processed prediction map alongside original
fig2 = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(tile)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(tile_prediction)
plt.axis("off")
fig2.suptitle("Processed prediction map", y=0.82)

plt.show()
