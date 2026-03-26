# puma-challenge-tissue-nuclei-segmentation
Research pipeline for tissue and nuclei segmentation on melanoma histopathology images for the PUMA Grand Challenge using nnU-Net and HoverNeXt models.

# System Environment - SOMAI Server 
Home Directory (Limited Storage)
Data Directory (Large Storage) 

All large files including:
* Dataset
* nnU-Net directories
* Model Output
* Virtual environments
* Scripts
were stored inside ~/data/

# Initial Setup 

# Create Virtual Environment
Created Python 3.9 environment for nnU-Net. All dependencies for the nnU-Net environment were saved into a requirements.txt file for reproducibility. 

# Created nnU-Net Directory Structure
Inside the data directory, I manually created the required nnU‑Net folder structure. 

Commands Used..
cd ~/data..

mkdir nnUNet_raw..
mkdir nnUNet_preprocessed..
mkdir nnUNet_results..

Directory Structure..
~/data/..
├── nnUNet_raw/..
├── nnUNet_preprocessed/..
├── nnUNet_results/..
These directories are required by nnU‑Net...

Environment Variables Set..
export nnUNet_raw=~/data/nnUNet_raw..
export nnUNet_preprocessed=~/data/nnUNet_preprocessed..
export nnUNet_results=~/data/nnUNet_results..

Added to .bashrc..
nano ~/.bashrc..

Added..
export nnUNet_raw=~/data/nnUNet_raw..
export nnUNet_preprocessed=~/data/nnUNet_preprocessed..
export nnUNet_results=~/data/nnUNet_results..

Then..
source ~/.bashrc.. 



