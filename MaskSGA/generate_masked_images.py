import os
import sys
import json
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def setup_dino(dino_repo_path):

    if not os.path.exists(dino_repo_path):
        os.system(f"git clone https://github.com/IDEACVR/DINO.git {dino_repo_path}")

    os.chdir(dino_repo_path)
    if dino_repo_path not in sys.path:
        sys.path.insert(0, dino_repo_path)


def load_dino_model(model_config_path, model_checkpoint_path, device="cuda"):

    from main import build_model_main
    from util.slconfig import SLConfig

    args = SLConfig.fromfile(model_config_path)
    args.device = device

    model, _, postprocessors = build_model_main(args)

    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    return model, postprocessors


def apply_transforms(pil_image):

    import datasets.transforms as DT

    target = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0)}
    img, tgt = DT.RandomResize([800], max_size=1333)(pil_image, target)
    img, tgt = DT.ToTensor()(img, tgt)
    img, tgt = DT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img, tgt)
    return img


# ──────────────────────────────────────────────
# STEP 1: GENERATE BOUNDING BOXES
# ──────────────────────────────────────────────

def generate_boxes(
    model,
    postprocessors,
    source_root,
    json_output_path,
    style_list,
    genders=("woman", "man"),
    person_class_id=1,
    sufficient_score=0.3,
    score_margin=0.3,
    device="cuda",
):

    os.makedirs(json_output_path, exist_ok=True)
    postprocess_size = torch.tensor([[1.0, 1.0]], device=device)  # boxes in [0,1] range

    for style in style_list:
        for gender in genders:
            folder_path = os.path.join(source_root, style, gender)
            output_file = os.path.join(json_output_path, f"{style}_{gender}_boxes.json")

            results = {}

            for image_name in sorted(os.listdir(folder_path)):
                image_path = os.path.join(folder_path, image_name)

                try:
                    pil_image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"Could not open {image_path}: {e}")
                    continue

                image = apply_transforms(pil_image)

                with torch.no_grad():
                    output_raw = model(image.to(device)[None])
                output = postprocessors["bbox"](output_raw, postprocess_size)[0]

                person_mask = output["labels"] == person_class_id
                person_scores = output["scores"][person_mask]

                if person_scores.numel() > 0:
                    top_score = person_scores.max()
                    min_score = min(sufficient_score, (1 - score_margin) * top_score)
                    selection_mask = person_mask & (output["scores"] > min_score)

                    results[image_name] = {
                        "scores": output["scores"][selection_mask].tolist(),
                        "boxes_xyxy": output["boxes"][selection_mask].tolist(),
                        "n_boxes": selection_mask.sum().item(),
                    }
                else:
                    print(f"No person found in {style}/{gender}/{image_name}")
                    results[image_name] = {
                        "scores": [],
                        "boxes_xyxy": [],
                        "n_boxes": 0,
                    }

            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

            print(f"Saved: {output_file}")



def mask_images_from_json(
    json_path,
    source_folder,
    destination_folder,
    mode,
    score_threshold=0.3,
    score_threshold_margin=0.3,
):

    if mode not in ("bg", "no_bg"):
        raise ValueError(f"mode must be 'bg' or 'no_bg', got {mode!r}")

    json_path = Path(json_path)
    source_folder = Path(source_folder)
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)

    if not source_folder.exists():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    with open(json_path, "r") as f:
        data = json.load(f)

    for image_name, info in data.items():
        image_path_orig = source_folder / image_name
        image_path_new = destination_folder / image_name

        if not image_path_orig.exists():
            print(f"Warning: {image_name} not found in source folder, skipping.")
            continue

        try:
            img_orig = Image.open(image_path_orig).convert("RGB")
            width, height = img_orig.size

            boxes = info.get("boxes_xyxy", [])
            scores = info.get("scores", [])

            if mode == "bg":
                img = img_orig.copy()
                rect_fill = (0, 0, 0)
            else:
                img = Image.new("RGB", (width, height), (0, 0, 0))
                rect_fill = (255, 255, 255)

            draw = ImageDraw.Draw(img)

            if info["n_boxes"] > 0:
                top_score = max(scores)
                image_threshold = min(score_threshold, (1 - score_threshold_margin) * top_score)

                for box, score in zip(boxes, scores):
                    if score < image_threshold:
                        continue

                    x1, y1, x2, y2 = box
                    x1 = max(0, int(x1 * width))
                    x2 = min(width, int(x2 * width))
                    y1 = max(0, int(y1 * height))
                    y2 = min(height, int(y2 * height))

                    draw.rectangle([x1, y1, x2, y2], fill=rect_fill)

            img.save(image_path_new)

        except Exception as e:
            print(f"Failed processing {image_name}: {e}")

    print("Masking completed successfully.")


def generate_masked_images(
    source_root,
    json_path,
    output_folder,
    style_list,
    genders=("man", "woman"),
    modes=("bg", "no_bg"),
    score_threshold=0.3,
    score_threshold_margin=0.3,
):

    for style in style_list:
        print(f"Processing style: {style}")

        for gender in genders:
            for mode in modes:
                mask_images_from_json(
                    json_path=os.path.join(json_path, f"{style}_{gender}_boxes.json"),
                    source_folder=os.path.join(source_root, style, gender),
                    destination_folder=os.path.join(output_folder, f"{style}_{gender}_{mode}"),
                    mode=mode,
                    score_threshold=score_threshold,
                    score_threshold_margin=score_threshold_margin,
                )
                print(f"  {style} / {gender} / {mode} — done.")



if __name__ == "__main__":

    DINO_REPO_PATH        = "add your path here"   # local path for the DINO repo
    MODEL_CONFIG_PATH     = "add your path here"   # DINO model config file (.py)
    MODEL_CHECKPOINT_PATH = "add your path here"   # DINO checkpoint file (.pth)

    SOURCE_ROOT           = "add your path here"   # root with style/gender image subfolders
    JSON_OUTPUT_PATH      = "add your path here"   # folder to save bounding box JSONs
    OUTPUT_FOLDER         = "add your path here"   # folder to save masked images

    STYLE_LIST            = ["add your styles here"]   # e.g. ["impressionism", "realism"]
    GENDERS               = ["woman", "man"]
    MODES                 = ["bg", "no_bg"]

    SUFFICIENT_SCORE      = 0.3
    SCORE_MARGIN          = 0.3
    DEVICE                = "cuda"

    # Step 1: Set up DINO and generate bounding boxes
    setup_dino(DINO_REPO_PATH)

    model, postprocessors = load_dino_model(
        model_config_path=MODEL_CONFIG_PATH,
        model_checkpoint_path=MODEL_CHECKPOINT_PATH,
        device=DEVICE,
    )

    generate_boxes(
        model=model,
        postprocessors=postprocessors,
        source_root=SOURCE_ROOT,
        json_output_path=JSON_OUTPUT_PATH,
        style_list=STYLE_LIST,
        genders=GENDERS,
        person_class_id=1,
        sufficient_score=SUFFICIENT_SCORE,
        score_margin=SCORE_MARGIN,
        device=DEVICE,
    )

    # Step 2: Generate masked images from the saved JSONs
    generate_masked_images(
        source_root=SOURCE_ROOT,
        json_path=JSON_OUTPUT_PATH,
        output_folder=OUTPUT_FOLDER,
        style_list=STYLE_LIST,
        genders=GENDERS,
        modes=MODES,
        score_threshold=SUFFICIENT_SCORE,
        score_threshold_margin=SCORE_MARGIN,
    )
