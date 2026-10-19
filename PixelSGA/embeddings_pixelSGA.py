import os
import gc
import random
import numpy as np
import torch
from PIL import Image
import open_clip
import argparse


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)



IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")

def is_image(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def load_model(model_name: str = "convnext_base_w", pretrained: str = "laion2b_s13b_b82k_augreg"):

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.cuda()
    model.eval()

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", model.context_length)
    print("Vocab size:", model.vocab_size)

    return model, preprocess


def encode_images(model, preprocess, folder_path: str, batch_size: int = 1) -> np.ndarray:

    img_paths = [
        os.path.join(folder_path, fname)
        for fname in sorted(os.listdir(folder_path))
        if is_image(fname)
    ]

    encoded_imgs = np.empty([0, 640])

    for _ in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[:batch_size]
        preprocessed = np.stack([
            preprocess(Image.open(p).convert("RGB")) for p in batch_paths
        ])
        with torch.no_grad():
            features = model.encode_image(
                torch.tensor(preprocessed, device="cuda")
            ).float().cpu().numpy()
        encoded_imgs = np.concatenate([encoded_imgs, features], axis=0)
        img_paths = img_paths[batch_size:]
        del preprocessed
        gc.collect()

    return encoded_imgs


def encode_pixel_resolutions(
    model,
    preprocess,
    base_input_path: str,
    output_path: str,
    resolutions: list,
    subres_prefix: str = "subres_",
):

    os.makedirs(output_path, exist_ok=True)

    for r in resolutions:
        print(f"\n── Resolution: {r} ──")
        subfolder_path = os.path.join(base_input_path, f"{subres_prefix}{r}")

        for sf in sorted(os.listdir(subfolder_path)):
            folder = os.path.join(subfolder_path, sf)
            if not os.path.isdir(folder):
                continue

            print(f"Processing: {folder}")
            features = encode_images(model, preprocess, folder)

            print(features[:2, 600:605])
            print(features[-2:, 600:605])

            out_file = os.path.join(output_path, f"{sf}.npy")
            np.save(out_file, features)
            print(f"Saved to: {out_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract OpenCLIP image embeddings for PixelSGA."
    )

    parser.add_argument(
        "--base_input_path",
        type=str,
        required=True,
        help="Root folder containing subres_* directories."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory where embedding .npy files will be saved."
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[224, 112, 56, 28, 14, 7, 4, 2, 1],
        help="Resolutions to process."
    )
    parser.add_argument(
        "--subres_prefix",
        type=str,
        default="subres_",
        help="Prefix used for resolution directories."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="convnext_base_w",
        help="OpenCLIP model architecture."
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s13b_b82k_augreg",
        help="OpenCLIP pretrained weights."
    )

    args = parser.parse_args()

    model, preprocess = load_model(
        model_name=args.model_name,
        pretrained=args.pretrained
    )

    encode_pixel_resolutions(
        model=model,
        preprocess=preprocess,
        base_input_path=args.base_input_path,
        output_path=args.output_path,
        resolutions=args.resolutions,
        subres_prefix=args.subres_prefix,
    )
