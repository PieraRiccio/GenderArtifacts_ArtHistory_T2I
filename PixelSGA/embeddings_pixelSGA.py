import os
import gc
import random
import argparse
import numpy as np
import torch
from PIL import Image
import open_clip
from transformers import AutoProcessor, SiglipVisionModel


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
    """
    Load the selected image embedding model.

    Supported models:
        clip: OpenCLIP
        siglip: Hugging Face SigLIP
    """

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

    elif embedding_model == "siglip":
        if model_name is None:
            model_name = "google/siglip-base-patch16-224"

        processor = AutoProcessor.from_pretrained(model_name)
        model = SiglipVisionModel.from_pretrained(model_name)
        model = model.to(DEVICE)
        model.eval()

        print(f"Loaded SigLIP model: {model_name}")

        return model, processor, DEVICE

    else:
        raise ValueError(
            f"Unsupported embedding model: {embedding_model}"
        )


def encode_images(
    model,
    preprocess,
    folder_path: str,
    embedding_model: str,
    device: str,
    batch_size: int = 1,
) -> np.ndarray:
    """
    Extract image embeddings from all supported images in a folder.
    """

    img_paths = [
        os.path.join(folder_path, fname)
        for fname in sorted(os.listdir(folder_path))
        if is_image(fname)
    ]

    if not img_paths:
        raise ValueError(f"No supported images found in {folder_path}")

    encoded_imgs = []

    for start in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[start:start + batch_size]

        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        with torch.no_grad():
            if embedding_model == "clip":
                preprocessed = torch.stack(
                    [preprocess(image) for image in images]
                ).to(device)

                features = model.encode_image(preprocessed).float()

            elif embedding_model == "siglip":
                inputs = preprocess(
                    images=images,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                outputs = model(**inputs)
                features = outputs.pooler_output.float()

            else:
                raise ValueError(
                    f"Unsupported embedding model: {embedding_model}"
                )

            encoded_imgs.append(features.cpu().numpy())

        del images
        gc.collect()

    return np.concatenate(encoded_imgs, axis=0)


def encode_pixel_resolutions(
    model,
    preprocess,
    base_input_path: str,
    output_path: str,
    resolutions: list,
    embedding_model: str,
    device: str,
    subres_prefix: str = "subres_",
    batch_size: int = 1,
):
    """
    Extract and save embeddings for each resolution and image folder.
    """

    os.makedirs(output_path, exist_ok=True)

    for resolution in resolutions:
        print(f"\n--- Resolution: {resolution} ---")

        subfolder_path = os.path.join(
            base_input_path,
            f"{subres_prefix}{resolution}",
        )

        if not os.path.isdir(subfolder_path):
            raise FileNotFoundError(
                f"Resolution directory not found: {subfolder_path}"
            )

        for folder_name in sorted(os.listdir(subfolder_path)):
            folder = os.path.join(subfolder_path, folder_name)

            if not os.path.isdir(folder):
                continue

            print(f"Processing: {folder}")

            features = encode_images(
                model=model,
                preprocess=preprocess,
                folder_path=folder,
                embedding_model=embedding_model,
                device=device,
                batch_size=batch_size,
            )

            output_file = os.path.join(
                output_path,
                f"{folder_name}.npy",
            )

            np.save(output_file, features)

            print(f"Embedding shape: {features.shape}")
            print(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract image embeddings for PixelSGA."
    )

    parser.add_argument(
        "--base_input_path",
        type=str,
        required=True,
        help="Root folder containing subres_* directories.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory where embedding .npy files will be saved.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        choices=["clip", "siglip"],
        required=True,
        help="Embedding model to use: clip or siglip.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[224, 112, 56, 28, 14, 7, 4, 2, 1],
        help="Resolutions to process.",
    )
    parser.add_argument(
        "--subres_prefix",
        type=str,
        default="subres_",
        help="Prefix used for resolution directories.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "Model architecture for CLIP or Hugging Face model ID "
            "for SigLIP. Uses the model-specific default if omitted."
        ),
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s13b_b82k_augreg",
        help="Pretrained weights identifier for OpenCLIP.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images to process per batch.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")

    model, preprocess, device = load_model(
        embedding_model=args.embedding_model,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )

    encode_pixel_resolutions(
        model=model,
        preprocess=preprocess,
        base_input_path=args.base_input_path,
        output_path=args.output_path,
        resolutions=args.resolutions,
        embedding_model=args.embedding_model,
        device=device,
        subres_prefix=args.subres_prefix,
        batch_size=args.batch_size,
    )
