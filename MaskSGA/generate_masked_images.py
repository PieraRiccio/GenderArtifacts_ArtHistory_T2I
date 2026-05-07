
"""
Code to generate transformed images for MaskSGA
"""


"""
Step I: Generate Boxes
"""

### prelims

# 1. Clone the DINO repo if not already done
import os, sys
if not os.path.exists('DINO'):
    os.system('git clone https://github.com/IDEACVR/DINO.git')

# 2. Change into the repo directory and add it to sys.path
os.chdir('DINO')
sys.path.insert(0, os.getcwd())

# 3. other imports
from main import build_model_main
from util.slconfig import SLConfig
import torch, json

import datasets.transforms as DT

def apply_transforms(pil_image):
    target = {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0)}
    img, tgt = DT.RandomResize([800], max_size=1333)(pil_image, target)
    img, tgt = DT.ToTensor()(img, tgt)
    img, tgt = DT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img, tgt)
    return img

# load DINO model
device = torch.device("cuda")

args = SLConfig.fromfile(model_config_path) # set model_config_path
args.device = 'cuda'

model, criterion, postprocessors = build_model_main(args)

checkpoint = torch.load(model_checkpoint_path, map_location="cpu") # set model_checkpoint_path
model.load_state_dict(checkpoint["model"])

model = model.to(device)
_ = model.eval()

postprocess_size = torch.tensor([[1.0, 1.0]], device=device) # will give box coordinates in [0,1] range

SUFFICIENT_SCORE = 0.3 # all boxes with at least this score will be considered
SCORE_MARGIN = 0.3 # all boxes wthin this percentage of the top score also





### loop through images and save a JSON per style

PERSON_CLASS_ID = 1 # coco names

for style in STYLE_LIST:
    for gender in ['woman', 'man']:

        # set source path
        folder_path = f"{SOURCE_ROOT}/{style}/{gender}/"
        output_file_path = f"{JSON_PATH}/{style}_{gender}_boxes.json"

        results = {}

        for image_name in os.listdir(folder_path):

            image_path = os.path.join(folder_path, image_name)
            ## load and prepare image
            pil_image = Image.open(image_path).convert('RGB')
            image = apply_transforms(pil_image)

            # run DINO-DETR
            with torch.no_grad():
              output_raw = model(image.to(device)[None])
            output = postprocessors['bbox'](output_raw, postprocess_size)[0]

            # select boxes
            person_mask = output['labels'] == PERSON_CLASS_ID
            person_scores = output['scores'][person_mask]
            # if there are person boxes, record as specified
            if person_scores.numel() > 0:
                top_score = person_scores.max()
                min_score = min(SUFFICIENT_SCORE, (1 - SCORE_MARGIN) * top_score)
                selection_mask = person_mask & (output['scores'] > min_score)
                # record results
                results[image_name] = {
                    "scores": output['scores'][selection_mask].tolist(), # n_boxes
                    "boxes_xyxy": output['boxes'][selection_mask].tolist(), # n_boxes x 4
                    "n_boxes": selection_mask.sum().item() # integer
                }
            # if no persons detected, record empty lists
            else:
                print(f'no person found in {style}/{gender}/{image_name}')
                results[image_name] = {
                    "scores": [], # list of length n_boxes
                    "boxes_xyxy": [], # nested list of shape n_boxes x 4
                    "n_boxes": 0 # integer
                }

        # Save one json per image folder (per style+gender)
        with open(output_file_path, "w") as f:
            json.dump(results, f, indent=4)

        print('saved: ', output_file_path)



"""
Step II: Generate Masked Images

"""

### prelims

# load packages
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import os

# function to generate images

def mask_images_from_json_pillow(
    json_path, source_folder, destination_folder,
    score_threshold=0.3, score_threshold_margin=0.3,
    mode="bg"  # "bg" | "no_bg"
):
    """
    Mask bounding boxes in images using JSON info.

    mode="bg"   : draw black rectangles over detected boxes on the original image.
    mode="no_bg": draw white rectangles over detected boxes on a black background.
    """
    if mode not in ("bg", "no_bg"):
        raise ValueError(f"mode must be 'bg' or 'no_bg', got {mode!r}")

    json_path = Path(json_path)
    source_folder = Path(source_folder)
    destination_folder = Path(destination_folder)

    if not source_folder.exists():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    with open(json_path, "r") as f:
        data = json.load(f)

    for image_name, info in data.items():

        image_path_orig = source_folder / image_name
        image_path_new = destination_folder / image_name

        if not image_path_orig.exists():
            print(f"Warning: {image_name} not found in folder, skipping.")
            continue

        try:
            img_orig = Image.open(image_path_orig).convert("RGB")
            width, height = img_orig.size

            boxes = info.get("boxes_xyxy", [])
            scores = info.get("scores", [])

            if mode == "bg":
                img = img_orig
                rect_fill = (0, 0, 0)
            else:  # no_bg
                img = Image.new("RGB", (width, height), (0, 0, 0))
                rect_fill = (255, 255, 255)

            draw = ImageDraw.Draw(img)

            if info['n_boxes'] > 0:
                top_score = max(scores)
                image_threshold = min(score_threshold, (1 - score_threshold_margin) * top_score)

                for box, score in zip(boxes, scores):
                    if score < image_threshold:
                        continue

                    x1, y1, x2, y2 = box

                    x1 = int(x1 * width);  x2 = int(x2 * width)
                    y1 = int(y1 * height); y2 = int(y2 * height)

                    x1, x2 = max(0, x1), min(width, x2)
                    y1, y2 = max(0, y1), min(height, y2)

                    draw.rectangle([x1, y1, x2, y2], fill=rect_fill)

            destination_folder.mkdir(parents=True, exist_ok=True)
            img.save(image_path_new)

        except Exception as e:
            print(f"Failed processing {image_name}: {e}")

    print("Masking completed successfully.")

### generate masked images with visible background

for style in STYLE_LIST:
    print(f'(processing {style}.)')

    for gender in ['man', 'woman']:
        mask_images_from_json_pillow(
            json_path =  f"{JSON_PATH}/{style}_{gender}_boxes.json",
            source_folder = f"{SOURCE_ROOT}/{style}/{gender}/",
            destination_folder = f"{OUTPUT_FOLDER}{style}_{gender}_bg/",
            score_threshold=0.3,
            score_threshold_margin=0.3,
            mode='bg'
        )
        print(style, gender, "done.")

### generate masked images with masked background

for style in STYLE_LIST:
    print(f'(processing {style}.)')

    for gender in ['man', 'woman']:
        mask_images_from_json_pillow(
                json_path =  f"{JSON_PATH}/{style}_{gender}_boxes.json",
                source_folder = f"{SOURCE_ROOT}/{style}/{gender}/",
            destination_folder = f"{OUTPUT_FOLDER}{style}_{gender}_no_bg/",
            score_threshold=0.3,
            score_threshold_margin=0.3,
            mode='no_bg'
        )
        print(style, gender, "done.")