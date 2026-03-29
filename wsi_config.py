import os
from pathlib import Path
import pandas as pd

#################################################################
### Functions and utilities
#################################################################
def norm_01(x_batch):  # Use this for models trained on 0-1 scaled data
    x_batch = x_batch / 255
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch

def z_norm(x_batch):  # use this for default nnUNet models, using z-score normalized data
    mean = x_batch.mean(axis=(-2, -1), keepdims=True)
    std = x_batch.std(axis=(-2, -1), keepdims=True)
    x_batch = ((x_batch - mean) / (std + 1e-8))
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch

def csv_to_matches(path_df):
    df = pd.read_csv(path_df)
    if len(df.columns) > 2:
        raise ValueError("The DataFrame should have only 2 columns: first contains image path and second contains mask path.")
    matches = list(df.itertuples(index=False, name=None))
    return matches

def return_matches_to_run(matches, output_folder):
    runtime_stems = [file[:-12] for file in os.listdir(output_folder) if file.endswith('_runtime.txt')]
    imgs, _ = zip(*matches)
    img_stems = [Path(file).stem for file in imgs]
    matches_to_run_idx = [i for i in range(len(img_stems)) if img_stems[i] not in runtime_stems]
    matches_to_run = [matches[i] for i in matches_to_run_idx]

    if len(matches_to_run) == 0:
        print(f"\nAll files have been processed already, see {output_folder}")
    else:
        print(f"\nReturning {len(matches_to_run)} matches that are not finished yet")
    return matches_to_run

#################################################################
### SET CONFIG
#################################################################

### TASK AND MODEL
# Base path to your trained nnUNet results
model_base_path = os.path.expanduser('~/data/nnUNet_results/Dataset300_Melanoma/nnUNetTrainer__nnUNetPlans__2d')

# Use norm_01 since your model is trained on 0-1 scaled data
norm = norm_01  
output_minus_1 = True  # subtract 1 from predictions because 0 = background

# List of checkpoint paths for each fold for ensembling
checkpoint_paths = [
    os.path.join(model_base_path, f'fold_{i}', 'checkpoint_best.pth')
    for i in range(5)
]

### OUTPUT FOLDER
output_folder = Path(os.path.expanduser('~/data/wsi_inference'))
os.makedirs(output_folder, exist_ok=True)

### MATCHES YOU WANT TO RUN
# Since you don't have a mask, set mask path to None
matches_to_run = [
    (os.path.expanduser('~/data/svs/TCGA-3N-A9WB-01Z-00-DX1.A9950ED4-9480-455C-AE0D-8E076D4DA432.svs'), None)
]

rerun_unfinished = True  # rerun unfinished slides
overwrite = True  # always process and overwrite outputs

### SAMPLING STUFF
spacing = 0.5  # high resolution
model_patch_size = 512
sampler_patch_size = 4 * model_patch_size  # 2048
cpus = 1

### WANDB
use_wandb = False

#################################################################
### USAGE NOTES
#################################################################

"""
To run WSI inference with this config:

python3 -u nnUNet_run_WSI_inference_REWORK_using_config.py YourConfigFileName

The script will:
- Load the best checkpoint from each fold for ensembling
- Process the single WSI located at ~/data/svs/...
- Save outputs to ~/data/wsi_inference
- Use 0-1 normalization and subtract 1 from predictions
- Run at 0.5 spacing with patch size 512
"""
