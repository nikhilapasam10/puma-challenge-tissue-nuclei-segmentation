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

nnUNet_raw (Dataset100_Melanoma), nnUNet_preprocessed, nnUNet_results  

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
Manually cross-checked nnU-Net tissue segmentation results for the ensembled model

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

---

# TIA Toolbox Experiments
Thinking of using the TIA Toolbox pretrained model (semantic segmentation) and then using nnU-Net to fine-tune on PUMA Dataset.  

Created virtual environment tia with necessary installations to run TIA Toolbox.  

Successfully produce visualized segmentations for 3 images using TIA Toolbox pretrained model (fcn_resnet50_unet-bcss).  

Attempted to manually create a function to computed class-based dice scores.  

Produced incorrect dice scores because class labels from GT and TIA output do not match.  

GT Mask:  
- Class 0: Background
- Class 1: Tumor
- Class 2: Stroma
- Class 3: Epidermis
- Class 4: Blood Vessel
- Class 5: Necrosis 

TIA ToolBox:  
- Class 0: Tumor
- Class 1: Stroma
- Class 2: Inflammatory
- Class 3: Necrosis
- Class 4: Other Regions

Researched and found TIA Toolbox has a built-in function to compute dice metrics.  

Evaluated TIAToolbox’s segmentation output for a single metastatic ROI by comparing its predicted label mask to the ground-truth (GT) annotation.  

Because TIAToolbox only predicts three meaningful classes that overlap with GT labels—tumor, stroma, and necrosis—first mapped TIAToolbox’s class indices into the GT label space:  
- TIAToolbox 0 → GT 1 (tumor)
- TIAToolbox 1 → GT 2 (stroma)
- TIAToolbox 3 → GT 5 (necrosis)

Classes that TIAToolbox does not model (GT classes 3: epidermis and 4: blood vessels) were handled separately.  

Because TIAToolbox lacks explicit classes for epidermis (GT 3) and blood vessels (GT 4), I tried to find out how often these regions were misclassified as TIAToolbox class 4 ("other").  

Added visualization outputs along with corresponding dice scores in one-drive.  

Results:  
- Poor segmentation
- Discarded
---

# nnU-Net Tissue Segmentation using Zenodo Pretrained Weights
Tissue segmentation inference on PUMA dataset images using the Zenodo pretrained weights.  

Created directories zenodo_images to run inference on sample pngs and zenodo_predictions to store the outputs. 

Copied checkpoint_best.pth to ~/data/nnunetv2/nnunetv2_hist/nnUNet_results/Dataset526_Mark/nnUNetTrainer_nnUNetPlans_2d/fold_4/  

Used the predict_from_modelfolder command and used the dataset.json from https://github.com/tueimage/PUMA-challenge-baseline-track2/tree/master/nnunetv2/nnunetv2_hist/nnUNet_raw/Dataset526_Mark.  

```
source ~/data/nnunet_env/bin/activate 

nnUNetv2_predict_from_modelfolder \
-i ~/data/zenodo_images \
-o ~/data/zenodo_predictions \
-m ~/data/nnunetv2/nnunetv2_hist/nnUNet_results/Dataset526_Mark/nnUNetTrainer_nnUNetPlans_2d \
-f 4 \
-chk checkpoint_best.pth 

nnUNetv2_evaluate_folder \
~/data/zenodo_labels \
~/data/zenodo_predictions \
-djfile ~/data/nnunetv2/nnunetv2_hist/nnUNet_results/Dataset526_Mark/nnUNetTrainer_nnUNetPlans_2d/dataset.json \
-pfile ~/data/nnunetv2/nnunetv2_hist/nnUNet_results/Dataset526_Mark/nnUNetTrainer_nnUNetPlans_2d/plans.json 
```

---

# HoverNext Nuclei Segmentation Using Zenodo Pretrained Weights (Track 1)

Cloned Repo  
https://github.com/mschuiveling/hover-next-inference-tils-melanoma  

Create Virtual Environment  
All dependencies were saved into a requirements.txt file for reproducibility.  

Make Directory for HoverNext Output  
```
mkdir ~/data/hovernext_class_inference
```

Run Inference  
Only need to change code for input and output directories  
```
source ~/data/hovernext_venv/bin/activate
python ~/data/main.py \
>     --input ~/data/zenodo_images/training_set_metastatic_roi_018_0000.png \ 
>     --output_root ~/data/hovernext_class_inference \
>     --cp ~/data/puma_convnextv2_base \
>     --save_polygon \
>     --metric f1
```

HoverNext Output  
Files generated:  
- class_inst.json: maps each instance → a class
- pinst_pp/:  instance ID per pixel (1…N nuclei)

Created nuclei segmentation map and corresponding legend by loading the pinst_pp Zarr data and mapping instance IDs from class_inst.json to colors for visualization. Used a fixed color scheme for the 3 nuclei classes.  

Code included in repository (hovernext_visualization.py).

---
# HoverNext Nuclei Segmentation Using Zenodo Pretrained Weights (Track 2)
Encountering an issue during inference  
While running the pipeline, I encountered a ValueError in post_process_utils.py, specifically:
“could not broadcast input array from shape (10, H, W) into shape (11, H, W).”  

From debugging, it appears that the post-processing code is expecting 11 output channels (10 nuclei classes + 1 background), whereas the model output only contains 10 channels corresponding to the nuclei classes. This mismatch is causing the failure during the post-processing step.  

NEXT STEPS:  
Check whether the toml file from the weights is correctly set  
<img width="1014" height="127" alt="image" src="https://github.com/user-attachments/assets/de4368c8-f1c0-420a-875a-85d9efc7c25c" />  

---

# WSI Inference  

Standard nnUNet is patch-based and therefore does not have any built-in function for us to run inference on the whole slide images.  

Found a paper that adapts nnUNet specifically for pathology applications by developing a WSI inference pipeline. Their approach extracts tissue-containing patches using a mask, applies sliding-window inference with Gaussian weighting and overlap, and then reconstructs full-slide segmentation masks. The pipeline also produces pixel-wise uncertainty maps, which help in identifying any incorrect predictions. The authors made their code implementation publicly available on GitHub (https://github.com/DIAGNijmegen/nnUNet-for-pathology/tree/nnunet_for_pathology_v2), and used the WholeSlideData library for better patch handling.  

This is the link to their paper: https://proceedings.mlr.press/v227/spronck24a.html.  
More information about the inference pipeline can be found in the Appendix A section  

Created a separate virtual environment specifically for Whole Slide Image (WSI) inference:
```
conda activate /data/npasam/nnunet_wsi_env
```
Cloned Repo  
https://github.com/DIAGNijmegen/nnUNet-for-pathology/tree/nnunet_for_pathology_v2  

Attempted to run WSI inference using the script nnUNetV2_run_WSI_inference_REWORK_with_config_newest.py and a config file I created (wsi_config.py).  
```
python -u pathology_code_and_utils/nnUNetV2_run_WSI_inference_REWORK_with_config.py /home/npasam/data/wsi_inference.py
```
Encountered issue during the preprocessing stage due to an import error related to crop_to_bbox. Specifically, nnunetv2/preprocessing/cropping/cropping.py contains the line:  
from acvl_utils.cropping_and_padding.bounding_boxes import crop_to_bbox  
In newer versions of acvl-utils (e.g., 0.2.1+), crop_to_bbox no longer appears to exist, which leads to the following error:
ImportError: cannot import name 'crop_to_bbox'

---

