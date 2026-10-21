import os
import gc
import random
import argparse
import numpy as np
import torch
from PIL import Image
import open_clip
from transformers import (
    AutoProcessor,
    SiglipVisionModel,
    AutoImageProcessor,
    Dinov2Model,
)


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def is_image(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def load_model(
    embedding_model: str,
    model_name: str = None,
    pretrained: str = "laion2b_s13b_b82k_augreg",
):
    """Load the selected embedding model and its image preprocessing."""
    if embedding_model == "clip":
        if model_name is None:
            model_name = "convnext_base_w"

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        model = model.to(DEVICE)
        model.eval()

        print(f"Loaded OpenCLIP model: {model_name}")
        print(f"Pretrained weights: {pretrained}")
        return model, preprocess, DEVICE

    if embedding_model == "siglip":
        if model_name is None:
            model_name = "google/siglip-base-patch16-224"

        processor = AutoProcessor.from_pretrained(model_name)
        model = SiglipVisionModel.from_pretrained(model_name).to(DEVICE)
        model.eval()

        print(f"Loaded SigLIP model: {model_name}")
        return model, processor, DEVICE

    if embedding_model == "dinov2":
        if model_name is None:
            model_name = "facebook/dinov2-with-registers-large"

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = Dinov2Model.from_pretrained(model_name).to(DEVICE)
        model.eval()

        print(f"Loaded DINOv2 model: {model_name}")
        print("Feature extraction: mean-pooled patch tokens")
        return model, processor, DEVICE

    raise ValueError(f"Unsupported embedding model: {embedding_model}")


def encode_images(
    model,
    preprocess,
    folder_path: str,
    embedding_model: str,
    device: str,
    batch_size: int = 1,
) -> np.ndarray:
    """Extract embeddings from supported images in one folder, in filename order."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Input image folder does not exist: {folder_path}")

    img_paths = [
        os.path.join(folder_path, filename)
        for filename in sorted(os.listdir(folder_path))
        if is_image(filename)
    ]
    if not img_paths:
        raise ValueError(f"No supported image files found in: {folder_path}")

    encoded_batches = []

    for start in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[start:start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        with torch.no_grad():
            if embedding_model == "clip":
                pixel_values = torch.stack(
                    [preprocess(image) for image in images]
                ).to(device)
                features = model.encode_image(pixel_values).float()

            elif embedding_model == "siglip":
                inputs = preprocess(images=images, return_tensors="pt")
                # SiglipVisionModel expects visual inputs only.
                pixel_values = inputs["pixel_values"].to(device)
                outputs = model(pixel_values=pixel_values)
                features = outputs.pooler_output.float()

            elif embedding_model == "dinov2":
                inputs = preprocess(images=images, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model(**inputs)

                # Exclude the CLS token and register tokens; pool patch tokens.
                num_register_tokens = getattr(
                    model.config, "num_register_tokens", 4
                )
                patch_features = outputs.last_hidden_state[
                    :, 1 + num_register_tokens:, :
                ]
                features = patch_features.mean(dim=1).float()

            else:
                raise ValueError(f"Unsupported embedding model: {embedding_model}")

            encoded_batches.append(features.cpu().numpy())

        del images
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return np.concatenate(encoded_batches, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Extract OpenCLIP, SigLIP, or DINOv2 embeddings from MaskSGA image folders."
    )
    parser.add_argument(
        "--input_man_bg", required=True,
        help="Folder containing male/man images with backgrounds masked.",
    )
    parser.add_argument(
        "--input_man_no_bg", required=True,
        help="Folder containing male/man images with only detected people marked.",
    )
    parser.add_argument(
        "--input_woman_bg", required=True,
        help="Folder containing female/woman images with backgrounds masked.",
    )
    parser.add_argument(
        "--input_woman_no_bg", required=True,
        help="Folder containing female/woman images with only detected people marked.",
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Directory in which the four .npy embedding files will be saved.",
    )
    parser.add_argument(
        "--embedding_model", choices=["clip", "siglip", "dinov2"],
        required=True, help="Embedding model to use: clip, siglip, or dinov2.",
    )
    parser.add_argument(
        "--model_name", default=None,
        help="Optional model identifier; defaults to the selected model's standard model.",
    )
    parser.add_argument(
        "--pretrained", default="laion2b_s13b_b82k_augreg",
        help="Pretrained weights identifier for OpenCLIP (ignored by SigLIP and DINOv2).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Number of images per batch; 1 is recommended for consistency with PixelSGA.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")

    input_folders = {
        ("bg", "male"): args.input_man_bg,
        ("no_bg", "male"): args.input_man_no_bg,
        ("bg", "female"): args.input_woman_bg,
        ("no_bg", "female"): args.input_woman_no_bg,
    }

    os.makedirs(args.output_path, exist_ok=True)
    model, preprocess, device = load_model(
        embedding_model=args.embedding_model,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )

    for (mode, gender_label), folder in input_folders.items():
        print(f"\nProcessing {gender_label} / {mode}: {folder}")
        features = encode_images(
            model=model,
            preprocess=preprocess,
            folder_path=folder,
            embedding_model=args.embedding_model,
            device=device,
            batch_size=args.batch_size,
        )

        output_file = os.path.join(args.output_path, f"{mode}_{gender_label}.npy")
        np.save(output_file, features)
        print(f"Saved embeddings with shape {features.shape} to: {output_file}")

    print("\nFinished extracting MaskSGA embeddings.")


if __name__ == "__main__":
    main()
