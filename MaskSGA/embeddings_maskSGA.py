import os
import gc
import random
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



IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")

def is_image(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def load_model(model_name: str = "convnext_base_w", pretrained: str = "laion2b_s13b_b82k_augreg"):
    """
    Loads and returns the OpenCLIP model, its transforms, and prints basic info.

    Args:
        model_name: OpenCLIP model architecture name.
        pretrained: Pretrained weights identifier.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.cuda()
    model.eval()

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", model.context_length)
    print("Vocab size:", model.vocab_size)

    return model, preprocess


def encode_images(model, preprocess, folder_path: str, batch_size: int = 1) -> np.ndarray:
    """
    Encodes all images in a folder using the given model.

    Args:
        model: OpenCLIP model (on CUDA).
        preprocess: Preprocessing transform for the model.
        folder_path: Folder containing images to encode.
        batch_size: Number of images per batch.

    Returns:
        Array of shape (N, embedding_dim).
    """
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


def encode_mask_modes(
    model,
    preprocess,
    base_input_path: str,
    output_path: str,
    styles: list,
    modes: list,
    genders: list,
    gender_map: dict,
    folder_structure_fn,
    output_filename_fn,
):

    for style in styles:
        for mode in modes:
            for gender in genders:
                folder = folder_structure_fn(base_input_path, style, mode, gender)
                gender_label = gender_map[gender]

                print(f"\nProcessing: {folder}")

                features = encode_images(model, preprocess, folder)

                print(features[:2, 600:605])
                print(features[-2:, 600:605])

                out_file = output_filename_fn(output_path, style, mode, gender_label)
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                np.save(out_file, features)
                print(f"Saved to: {out_file}")



if __name__ == "__main__":

    BASE_INPUT_PATH = "add your path here"        # root folder with image data
    OUTPUT_PATH     = "add your path here"        # root folder for .npy output
    STYLES          = ["add your styles here"]    # e.g. ["NEUTRAL", "realism"]
    MODES           = ["bg", "no_bg"]             # background modes
    GENDERS         = ["man", "woman"]            # gender subfolder names
    GENDER_MAP      = {"man": "male", "woman": "female"}

    # Define how image folders are structured.
    # Adjust to match your folder layout.
    def folder_structure_fn(base_input_path, style, mode, gender):
        return os.path.join(base_input_path, style, mode, gender)

    # Define how output .npy filenames are constructed.
    # Adjust to match your naming convention.
    def output_filename_fn(output_path, style, mode, gender_label):
        return os.path.join(output_path, style, f"{style}_{mode}_{gender_label}.npy")
    # ─────────────────────────────────────────────────────────────────

    model, preprocess = load_model()

    encode_mask_modes(
        model=model,
        preprocess=preprocess,
        base_input_path=BASE_INPUT_PATH,
        output_path=OUTPUT_PATH,
        styles=STYLES,
        modes=MODES,
        genders=GENDERS,
        gender_map=GENDER_MAP,
        folder_structure_fn=folder_structure_fn,
        output_filename_fn=output_filename_fn,
    )
