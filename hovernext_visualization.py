import zarr, json, numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Change appropriate directory paths
zip_path = '~/data/zenodo_nuclei_inference/training_set_primary_roi_024_0000/pinst_pp.zip'
json_path = '~/data/zenodo_nuclei_inference/training_set_primary_roi_024_0000/class_inst.json'
output_root = '~/data/hovernext_3_fixed'
zip_path = os.path.expanduser(zip_path)
json_path = os.path.expanduser(json_path)
output_root = os.path.expanduser(output_root)
os.makedirs(output_root, exist_ok=True)

output_combined = os.path.join(output_root, 'training_set_primary_roi_024_0000_mask_with_legend.png')

# --- Hardcoded color scheme ---
class_to_color = {
    0: [0,0,0],       # background
    1: [255,0,0],     # red
    2: [0,255,0],     # green
    3: [0,0,255]      # blue
}

# --- Load segmentation ---
store = zarr.storage.ZipStore(zip_path, mode='r')
z = zarr.open(store, mode='r')
seg_array = z[...]
if seg_array.ndim==3: seg_array = seg_array[0]

# --- Load instance->class mapping ---
with open(json_path,'r') as f:
    instance_to_class_raw = json.load(f)
instance_to_class = {int(inst):(int(val[0]) if isinstance(val,list) else int(val))
                     for inst,val in instance_to_class_raw.items()}

# --- Create class array ---
class_array = np.zeros_like(seg_array,dtype=np.uint8)
for inst, cls in instance_to_class.items(): class_array[seg_array==inst]=cls

# --- Create mask using HARD-CODED colors ---
H,W=class_array.shape
mask_rgb=np.zeros((H,W,3),dtype=np.uint8)
for cls,color in class_to_color.items(): mask_rgb[class_array==cls]=color
mask_img=Image.fromarray(mask_rgb)

# --- Create legend ---
legend_height = 50+30*len(class_to_color)
legend_width = 200
legend_img=Image.new('RGB',(legend_width,legend_height),(255,255,255))
draw=ImageDraw.Draw(legend_img)
try: font=ImageFont.truetype("DejaVuSans-Bold.ttf",16)
except: font=ImageFont.load_default()
y=10
for cls,color in class_to_color.items():
    draw.rectangle([10,y,30,y+20],fill=tuple(color))
    draw.text((40,y),f'Class {cls}',fill=(0,0,0),font=font)
    y+=30

# --- Combine mask + legend ---
combined_width=W+legend_width
combined_height=max(H,legend_height)
combined_img=Image.new('RGB',(combined_width,combined_height),(255,255,255))
combined_img.paste(mask_img,(0,0))
combined_img.paste(legend_img,(W,0))
combined_img.save(output_combined)
print(f"Saved combined mask + legend at {output_combined}")
