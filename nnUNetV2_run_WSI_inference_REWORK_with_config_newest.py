#################################################################
### NNUNET WSI INFERENCE WITH HALF OVERLAP
#################################################################
### Compatible with acvl-utils 0.2.1
#################################################################

import os
import sys
import importlib
from pathlib import Path
import numpy as np
import torch
import time
import tifffile
import wandb

from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators.patchiterator import create_patch_iterator
from wholeslidedata.buffer.patchcommander import PatchConfiguration
from wholeslidedata.samplers.utils import crop_data

from nnunetv2.utilities.file_path_utilities import load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

#################################################################
### CONFIG IMPORT
#################################################################

def import_config(identifier):
    identifier_path = Path(identifier)
    if identifier_path.suffix == ".py" or identifier_path.is_absolute() or len(identifier_path.parts) > 1:
        config_path = identifier_path.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        base_dir = str(config_path.parent)
        module_name = config_path.stem
    else:
        base_dir = str(Path(__file__).resolve().parent / "inference_configs")
        module_name = identifier

    if base_dir not in sys.path:
        sys.path.append(base_dir)

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Config module '{module_name}' not found in '{base_dir}'"
        )

if len(sys.argv) != 2 and len(sys.argv) != 8:
    print("\nINCORRECT FUNCTION CALL: \nPlease provide a config stem as argument and optionally input/output paths")
    sys.exit(1)

config_module_name = sys.argv[1]
config = import_config(config_module_name)

model_base_path = config.model_base_path
norm = config.norm 
output_minus_1 = config.output_minus_1

output_img, output_mask = (None, None) if len(sys.argv) == 2 else (sys.argv[4], sys.argv[5])
output_folder = config.output_folder if len(sys.argv) == 2 else Path(output_img).parent
local_output_folder = Path('/tmp/workdir')
os.makedirs(output_folder, exist_ok=True)
os.makedirs(local_output_folder, exist_ok=True)

matches_to_run = config.matches_to_run if len(sys.argv) == 2 else [(sys.argv[2], sys.argv[3])]
rerun_unfinished = config.rerun_unfinished
overwrite = config.overwrite
spacing = config.spacing
model_patch_size = config.model_patch_size
sampler_patch_size = config.sampler_patch_size
cpus = getattr(config, 'cpus', 1)

use_wandb = config.use_wandb
if use_wandb:
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    if wandb_api_key is None:
        print("WANDB_API_KEY not found. Aborting...")
        sys.exit(1)

#################################################################
### UTILITIES
#################################################################

current_os = "w" if os.name == "nt" else "l"
def convert_path(path, to=current_os):
    if to in ["w", "win", "windows"]:
        path = path.replace("/data/pathology", "Z:")
        path = path.replace("/data/pa_cpgarchive1", "W:")
        path = path.replace("/data/pa_cpgarchive2", "X:")
        path = path.replace("/data/pa_cpg", "Y:")
        path = path.replace("/data/temporary", "T:")
        path = path.replace("/", "\\")
    if to in ["u", "unix", "l", "linux"]:
        path = path.replace("Z:", "/data/pathology")
        path = path.replace("W:", "/data/pa_cpgarchive1")
        path = path.replace("X:", "/data/pa_cpgarchive2")
        path = path.replace("Y:", "/data/pa_cpg")
        path = path.replace("T:", "/data/temporary")
        path = path.replace("\\", "/")
    return path

def ensemble_softmax_list(x_batch):
    logits_list = predictor.get_logits_list_from_preprocessed_data(torch.tensor(x_batch, dtype=torch.float32))
    softmax_list = [predictor.label_manager.apply_inference_nonlin(logits).numpy() for logits in logits_list]
    return softmax_list

def array_to_formatted_tensor(array):
    array = array.transpose(1, 0, 2, 3)
    return torch.tensor(array)

def softmax_list_and_mean_to_uncertainty(softmax_list, softmax_mean):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    uncertainty_loss_per_pixel_list = []
    for softmax in softmax_list:
        log_softmax = np.log(softmax + 1e-8)
        uncertainty_loss_per_pixel = loss(array_to_formatted_tensor(log_softmax),
                                          array_to_formatted_tensor(softmax_mean))
        uncertainty_loss_per_pixel_list.append(uncertainty_loss_per_pixel)
    uncertainty = torch.cat(uncertainty_loss_per_pixel_list).mean(dim=0)
    return uncertainty

def get_trim_indexes(y_batch):
    y = y_batch[0]
    r_is_empty = [not y[start:end].any() for start, end in zip(half_patch_size_start_idxs, half_patch_size_end_idxs)]
    c_is_empty = [not y[:, start:end].any() for start, end in zip(half_patch_size_start_idxs, half_patch_size_end_idxs)]

    if not any(r_is_empty) and not any(c_is_empty):
        return 0, y.shape[0], 0, y.shape[1]

    empty_rs_top = sum(r_is_empty)
    empty_rs_bottom = sum(r_is_empty[::-1])
    empty_cs_left = sum(c_is_empty)
    empty_cs_right = sum(c_is_empty[::-1])

    trim_top_idx = half_patch_size_start_idxs[max(empty_rs_top-1,0)]
    trim_bottom_idx = half_patch_size_end_idxs[::-1][max(empty_rs_bottom-1,0)]
    trim_left_idx = half_patch_size_start_idxs[max(empty_cs_left-1,0)]
    trim_right_idx = half_patch_size_end_idxs[::-1][max(empty_cs_right-1,0)]

    return trim_top_idx, trim_bottom_idx, trim_left_idx, trim_right_idx

def decode_buffer_states(state_array, cpus):
    state_mappings = {'FREE': 1, 'AVAILABLE': 2, 'RESERVED': 3, 'PROCESSING': 4}
    state_count = {state: np.sum(state_array == state_mappings[state]) for state in state_mappings}
    sum_states = sum(state_count.values())
    if state_count['AVAILABLE'] + state_count['PROCESSING'] == sum_states:
        message = f', iterator buffer saturated. Using {cpus} CPUs'
    elif state_count['AVAILABLE'] <= 1:
        message = f', iterator buffer empty or almost empty. Using {cpus} CPUs'
    else: 
        message = ''
    return state_count, message

def get_closest_value(value):
    return min([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64], key=lambda x:abs(x-value))

#################################################################
### LOAD MODEL
#################################################################

print('\nModel path:', model_base_path)
plans_dict = load_json(os.path.join(model_base_path, 'plans.json'))
dataset_dict = load_json(os.path.join(model_base_path, 'dataset.json'))

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_gpu=True,
    device=torch.device('cuda', 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=False
)
predictor.initialize_from_trained_model_folder(
    model_base_path,
    use_folds=(0,1,2,3,4),
    checkpoint_name='checkpoint_best.pth',
)

#################################################################
### AUTO CONFIG
#################################################################
half_model_patch_size = model_patch_size // 2
assert sampler_patch_size % half_model_patch_size == 0
output_patch_size = sampler_patch_size - 2 * half_model_patch_size
sampler_patch_size_range = list(range(sampler_patch_size))
half_patch_size_start_idxs = sampler_patch_size_range[0::half_model_patch_size]
half_patch_size_end_idxs = [idx + half_model_patch_size for idx in half_patch_size_start_idxs]

#################################################################
### WANDB INIT
#################################################################
if use_wandb:
    import datetime
    date = datetime.datetime.now().strftime("%Y%m%d")
    wandb.login(key=wandb_api_key)
    wandb.init(project='nnUNet_inference_checks', name=f'sample: {sampler_patch_size} patch: {model_patch_size} date: {date}')

#################################################################
### LOOP
#################################################################
print('\n####### START OF LOOP #######')
for idx_match, (image_path, mask_path) in enumerate(matches_to_run):
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    print(f'\n[NEXT MATCH] [{idx_match}/{len(matches_to_run)}]: {image_path}, {mask_path}')
    
    wsm_path = output_folder / (image_path.stem + '_nnunet.tif')
    wsu_path = output_folder / (image_path.stem + '_uncertainty.tif')
    
    if not overwrite:
        if os.path.isfile(output_folder / (image_path.stem + '_runtime.txt')):
            print(f'[SKIPPING] {image_path.stem} already processed')
            continue
        if not rerun_unfinished and os.path.isfile(wsm_path) and os.path.isfile(wsu_path):
            print(f'[SKIPPING] {image_path.stem} already processed')
            continue
    
    open(wsm_path, 'w').close()
    open(wsu_path, 'w').close()
    
    print(f'[RUNNING] {image_path.stem}')

    with WholeSlideImage(image_path, backend='openslide') as wsi:
        shape = wsi.shapes[wsi.get_level_from_spacing(spacing)]
        real_spacing = wsi.get_real_spacing(spacing)
        downsampling = get_closest_value(wsi.get_downsampling_from_spacing(spacing))
        offset = int((output_patch_size // 2) * downsampling)
    
    patch_configuration = PatchConfiguration(
        patch_shape=(sampler_patch_size,sampler_patch_size,3),
        spacings=(spacing,),
        overlap=(model_patch_size,model_patch_size),
        offset=(int(offset), int(offset)),
        center=True,
        write_shape=(output_patch_size, output_patch_size)
    )
    
    # Full arrays to write tiles into
    wsm_array = np.zeros(shape, dtype=np.uint8)
    wsu_array = np.zeros(shape, dtype=np.uint8)

    start_time = time.time()
    print('\nInitiating iterator...')
    
    with create_patch_iterator(
        image_path=image_path,
        mask_path=mask_path,
        patch_configuration=patch_configuration,
        backend='openslide',
        cpus=cpus
    ) as patch_iterator:

        for idx_batch, (x_batch, y_batch, info) in enumerate(patch_iterator):
            print(f'\t[processing tile {idx_batch}/{len(patch_iterator)}] ...', flush=True)
            
            x_batch = x_batch[0]
            y_batch = y_batch[0]
            trim_top_idx, trim_bottom_idx, trim_left_idx, trim_right_idx = get_trim_indexes([y_batch])
            
            x_batch_maybe_trimmed = x_batch[:, trim_top_idx:trim_bottom_idx, trim_left_idx:trim_right_idx, :]
            prep = norm(x_batch_maybe_trimmed)
            
            softmax_list = ensemble_softmax_list(prep)
            softmax_mean = np.array(softmax_list).mean(0)
            pred_output_maybe_trimmed = softmax_mean.argmax(axis=0) - (1 if output_minus_1 else 0)
            
            uncertainty = softmax_list_and_mean_to_uncertainty(softmax_list, softmax_mean)
            uncertainty_output_maybe_trimmed = np.array((uncertainty.clip(0,4)/4*255).int())
            
            # Reconstruct full tile
            pred_output = np.zeros((sampler_patch_size, sampler_patch_size))
            pred_output[trim_top_idx:trim_bottom_idx, trim_left_idx:trim_right_idx] = pred_output_maybe_trimmed
            uncertainty_output = np.zeros((sampler_patch_size, sampler_patch_size))
            uncertainty_output[trim_top_idx:trim_bottom_idx, trim_left_idx:trim_right_idx] = uncertainty_output_maybe_trimmed
            
            # --- replace crop_to_bbox with crop_data ---
            pred_output_inner = crop_data(pred_output, [output_patch_size, output_patch_size])
            uncertainty_output_inner = crop_data(uncertainty_output, [output_patch_size, output_patch_size])
            y_batch_inner = crop_data(y_batch, [output_patch_size, output_patch_size]).astype('int64')
            
            x_coord, y_coord = info['x']//downsampling, info['y']//downsampling
            x_coord -= output_patch_size//2
            y_coord -= output_patch_size//2
            
            # Write tiles into full arrays
            wsm_array[y_coord:y_coord+output_patch_size, x_coord:x_coord+output_patch_size] = pred_output_inner * y_batch_inner
            wsu_array[y_coord:y_coord+output_patch_size, x_coord:x_coord+output_patch_size] = uncertainty_output_inner * y_batch_inner

            if use_wandb and idx_batch % 10 == 0:
                state_array = patch_iterator._buffer_factory.buffer_state_memory.get_state_buffer()
                state_count, message = decode_buffer_states(state_array, cpus)
                print(f'\t\tBUFFER STATES (batch {idx_batch}): {state_count}, {message}', flush=True)

    # Save runtime
    run_time = time.time() - start_time
    with open(output_folder / (image_path.stem + '_runtime.txt'), "w") as f:
        f.write(str(run_time))

    # Save full masks as TIFF
    tifffile.imwrite(wsm_path, wsm_array)
    tifffile.imwrite(wsu_path, wsu_array)
    
    print(f'[COMPLETED] {image_path.stem} masks saved.\n')

print('[ALL DONE]')
sys.exit(0)