# PUMA Grand Challenge — Tissue & Nuclei Segmentation Pipeline

Research pipeline for tissue and nuclei segmentation on melanoma histopathology images for the PUMA Grand Challenge using nnU-Net and HoverNeXt models. 

---

# System Environment

Home Directory (Limited Storage)
Data Directory (Large Storage)

All large files including:
• Dataset
• Model outputs
• Virtual environments
• nnU-Net directories
• Scripts

were stored inside:

```
~/data/
```

---

# Initial Setup

## Create Virtual Environment

Created Python 3.9 environment for nnU-Net. All dependencies for the nnU-Net environment were saved into a requirements.txt file for reproducibility.

---

# Created nnU‑Net Directory Structure

Inside the data directory, I manually created the required nnU‑Net folder structure.

Commands Used

```
cd ~/data

mkdir nnUNet_raw
mkdir nnUNet_preprocessed
mkdir nnUNet_results
```

Directory Structure

```
~/data/
├── nnUNet_raw/
├── nnUNet_preprocessed/
├── nnUNet_results/
```

These directories are required by nnU‑Net.

Environment Variables Set

```
export nnUNet_raw=~/data/nnUNet_raw
export nnUNet_preprocessed=~/data/nnUNet_preprocessed
export nnUNet_results=~/data/nnUNet_results
```

Added to .bashrc

```
nano ~/.bashrc
```

Added

```
export nnUNet_raw=~/data/nnUNet_raw
export nnUNet_preprocessed=~/data/nnUNet_preprocessed
export nnUNet_results=~/data/nnUNet_results
```

Then

```
source ~/.bashrc
```

---

# Dataset Creation

Created Dataset Directory

```
cd ~/data/nnUNet_raw
mkdir Dataset100_Melanoma
cd Dataset100_Melanoma
```

Created Required Folders

```
mkdir imagesTr
mkdir labelsTr
mkdir imagesTs
```

Final Structure

```
Dataset100_Melanoma
├── imagesTr
├── labelsTr
├── imagesTs
└── dataset.json
```

---

# PUMA Dataset Details

• 103 Primary Cases
• 103 Metastatic Cases
• Total: 206 images

ROI Size

1024 × 1024

Context Images

5120 × 5120

Annotations - GeoJSON files

---

# Annotation Conversion

Challenge organizers converted GeoJSON annotations into segmentation masks

Cross‑checked 30 images manually

Uploaded labels Tr into shared One-Drive

---

# First Test Training (10 Images)

Created Small Dataset (Dataset100_Melanoma)

• 10 Training Images
• 2‑3 Test Images

Format needed for nnU-Net model: imagesTr, imagesTs, labelsTr, dataset.json 

imagesTr: images used for training (png), labelsTr: segmentation labels for the training images (png), imagesTs: images used for testing (png) 

Modified dataset.json so it would train for only 10 images 

NnUNet_raw (Dataset100_Melanoma), nnUNet_preprocessed, nnUNet_results 

Trained the test model for 20 epochs for 5 folds  

---

# Training nnU-Net

• 165 training images
• 41 testing images

During training, nnU-Net automatically generates a progress.png image and detailed log files to monitor progress.

Command Used

```
nnUNetv2_train Dataset200_Melanoma 2d nnUNetTrainer_250epochs nnUNetPlans
```

Training Configuration

• 5 fold cross validation
• 250 epochs
• Patch based training

Output Saved To

```
~/data/nnUNet_results
```

---

# Ensemble Model

Create directory to save predictions
Ran ensembling inference using each fold's best checkpoint file

Command

```
nnUNetv2_predict \
-i nnUNet_raw/Dataset200_Melanoma/imagesTs \
-o nnUNet_predictions \
-d Dataset200_Melanoma \
-c 2d \
--use_folds 0 1 2 3 4 \
--chk checkpoint_best.pth \
--save_probabilities
```

---

# Evaluation

```
nnUNetv2_evaluate_folder [ground_truth_folder] [prediction_folder]
```

nnUNetv2_evaluate_folder \
nnUNet_raw/Dataset200_Melanoma/labelsTs \
nnUNet_predictions
```

Generated:

- summary.json – contains all evaluation metrics in JSON format, including:
  - Overall metrics (foreground_mean) for all non-background classes (Dice, IoU, TP, FP, FN, TN).
  - Per-class averages (mean) with Dice, IoU, TP, FP, FN, TN, predicted voxels, reference voxels.
  - Per-case metrics (metric_per_case) showing detailed Dice, IoU, and counts for each image and class.
  - Dice scores – per class and per image, useful for analyzing performance on individual labels.
  - TP / FP / FN / TN counts – raw counts of prediction vs. ground truth voxels.
  - n_pred / n_ref – total predicted voxels and reference voxels for each class
---

# Training, Ensembling, Evaluating nnu-Net on Entire PUMA Dataset
• 206 training images (Dataset 300_Melanoma)

# TIA Toolbox Experiments

Created Environment

```
conda create -n tia python=3.9
conda activate tia
```

Installed

```
pip install tiatoolbox
```

Model Used

fcn_resnet50_unet-bcss

Results

Poor segmentation

Discarded

---

# HoverNext Nuclei Segmentation

Cloned Repo

```
git clone https://github.com/digitalpathologybern/hover_next_inference
```

Environment

```
conda create -n hovernext python=3.8
conda activate hovernext
```

Installed

```
pip install torch
pip install zarr
pip install toml
pip install tqdm
pip install segmentation_models_pytorch
```

Downgrade numpy

```
pip install numpy==1.26
```

Run Inference

```
python main.py \
--input images/*.png \
--output_root results \
--cp pretrained_model \
--only_inference
```

---

# WSI Inference

Used

nnUNet‑for‑pathology

Removed WholeSlideMaskWriter

Used

```
tifffile.imwrite
```

---

# Data Augmentation

Used

nnUNetTrainerDA5

Includes

• Rotation
• Scaling
• Mirroring
• Noise
• Blur

---

# Pretrained Models

Zenodo Weights

Command

```
nnUNetv2_predict_from_modelfolder
```

---

# Final Pipeline

Tissue Segmentation

nnU‑Net

Nuclei Segmentation

HoverNext

WSI Inference

nnUNet‑for‑pathology

---

# Challenges

• Dependency conflicts
• Disk quota
• NumPy version mismatch
• WSI memory constraints

---

# Future Work

• Improve nuclei segmentation
• Improve WSI inference
• Ensemble tissue + nuclei

---

# Author

Nikhila Pasam
PUMA Grand Challenge
Emory University

---

