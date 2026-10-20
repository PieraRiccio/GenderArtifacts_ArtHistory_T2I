import os
import gc
import random
import argparse
import numpy as np
import torch
from PIL import Image
import open_clip


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"
)


def is_image(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def load_model(
    model_name: str = "convnext_base_w",
    pretrained: str = "laion2b_s13b_b82k_augreg",
):
    """Load the OpenCLIP model and its image preprocessing transform."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.cuda()
    model.eval()

    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )
    print("Context length:", model.context_length)
    print("Vocab size:", model.vocab_size)

    return model, preprocess


def encode_images(model, preprocess, folder_path: str, batch_size: int = 1) -> np.ndarray:
    """
    Encode images in one folder.

    Images are processed in sorted filename order. The default batch size is 1,
    consistent with the project's requirement to avoid observed batch-dependent
    artifacts.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Input image folder does not exist: {folder_path}")

    img_paths = [
        os.path.join(folder_path, fname)
        for fname in sorted(os.listdir(folder_path))
        if is_image(fname)
    ]

    if not img_paths:
        raise ValueError(f"No supported image files found in: {folder_path}")

    encoded_batches = []

    for start in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[start : start + batch_size]

        preprocessed = np.stack([
            preprocess(Image.open(path).convert("RGB"))
            for path in batch_paths
        ])

        with torch.no_grad():
            features = model.encode_image(
                torch.tensor(preprocessed, device="cuda")
            ).float().cpu().numpy()

        encoded_batches.append(features)
        del preprocessed
        gc.collect()

    return np.concatenate(encoded_batches, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Extract OpenCLIP embeddings from pre-generated MaskSGA image folders."
    )
    parser.add_argument("--input_man_bg", required=True,
                        help="Folder containing male/man images with backgrounds masked.")
    parser.add_argument("--input_man_no_bg", required=True,
                        help="Folder containing male/man images with only detected people marked.")
    parser.add_argument("--input_woman_bg", required=True,
                        help="Folder containing female/woman images with backgrounds masked.")
    parser.add_argument("--input_woman_no_bg", required=True,
                        help="Folder containing female/woman images with only detected people marked.")
    parser.add_argument("--output_path", required=True,
                        help="Root folder in which the .npy embedding files will be saved.")
    parser.add_argument("--model_name", default="convnext_base_w",
                        help="OpenCLIP model architecture.")
    parser.add_argument("--pretrained", default="laion2b_s13b_b82k_augreg",
                        help="OpenCLIP pretrained weights identifier.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images per batch; use 1 to avoid observed batch-dependent artifacts.")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1.")

    # The input folders are supplied directly; no assumed style/mode/gender
    # directory hierarchy is required.
    input_folders = {
        ("bg", "male"): args.input_man_bg,
        ("no_bg", "male"): args.input_man_no_bg,
        ("bg", "female"): args.input_woman_bg,
        ("no_bg", "female"): args.input_woman_no_bg,
    }

    model, preprocess = load_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
    )

    for (mode, gender_label), folder in input_folders.items():
        print(f"\nProcessing {gender_label} / {mode}: {folder}")
        features = encode_images(
            model=model,
            preprocess=preprocess,
            folder_path=folder,
            batch_size=args.batch_size,
        )

        output_file = os.path.join(
            args.output_path,
            f"{mode}_{gender_label}.npy",
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, features)
        print(f"Saved embeddings with shape {features.shape} to: {output_file}")


if __name__ == "__main__":
    main()
