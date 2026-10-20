import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


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

    if not source_folder.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Detection JSON not found: {json_path}. "
            "This script does not run DINO detection; provide a JSON generated "
            "by the detector."
        )

    with json_path.open("r", encoding="utf-8") as f:
        detections = json.load(f)

    supported_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
    }

    for image_path in sorted(source_folder.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in supported_extensions:
            continue

        info = detections.get(image_path.name)
        if info is None:
            print(f"Warning: no detection entry for {image_path.name}; skipping.")
            continue

        try:
            with Image.open(image_path) as opened:
                original = opened.convert("RGB")

            width, height = original.size
            if mode == "bg":
                output = original.copy()
                rectangle_fill = (0, 0, 0)
            else:
                output = Image.new("RGB", (width, height), (0, 0, 0))
                rectangle_fill = (255, 255, 255)

            draw = ImageDraw.Draw(output)
            boxes = info.get("boxes_xyxy", [])
            scores = info.get("scores", [])

            if boxes and scores:
                top_score = max(scores)
                threshold = min(
                    score_threshold,
                    (1.0 - score_threshold_margin) * top_score,
                )

                for box, score in zip(boxes, scores):
                    if score < threshold:
                        continue
                    if len(box) != 4:
                        print(f"Warning: invalid box for {image_path.name}: {box}")
                        continue

                    x1, y1, x2, y2 = box
                    x1 = max(0, min(width, round(x1 * width)))
                    y1 = max(0, min(height, round(y1 * height)))
                    x2 = max(0, min(width, round(x2 * width)))
                    y2 = max(0, min(height, round(y2 * height)))

                    if x2 > x1 and y2 > y1:
                        draw.rectangle([x1, y1, x2, y2], fill=rectangle_fill)

            output.save(destination_folder / image_path.name)

        except Exception as exc:
            print(f"Failed processing {image_path.name}: {exc}")

    print(f"Saved {mode} images to: {destination_folder}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create MaskSGA masked images for separate female and male folders "
            "using precomputed person-detection JSON files. This script does "
            "not load DINO or perform detection."
        )
    )
    parser.add_argument("--input_female", required=True,
                        help="Folder containing female source images.")
    parser.add_argument("--input_male", required=True,
                        help="Folder containing male source images.")
    parser.add_argument("--boxes_female", required=True,
                        help="Precomputed detection JSON for female images.")
    parser.add_argument("--boxes_male", required=True,
                        help="Precomputed detection JSON for male images.")
    parser.add_argument("--output_base", required=True,
                        help="Directory where masked image folders will be saved.")
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--score_margin", type=float, default=0.3)
    args = parser.parse_args()

    input_folders = {
        "woman": Path(args.input_female).expanduser().resolve(),
        "man": Path(args.input_male).expanduser().resolve(),
    }
    json_paths = {
        "woman": Path(args.boxes_female).expanduser().resolve(),
        "man": Path(args.boxes_male).expanduser().resolve(),
    }
    output_base = Path(args.output_base).expanduser().resolve()

    for gender, folder in input_folders.items():
        if not folder.is_dir():
            parser.error(f"{gender} input folder does not exist: {folder}")
    for gender, json_path in json_paths.items():
        if not json_path.is_file():
            parser.error(f"{gender} detection JSON does not exist: {json_path}")
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score_threshold must be between 0 and 1.")
    if not 0 <= args.score_margin <= 1:
        parser.error("--score_margin must be between 0 and 1.")

    for gender in ("woman", "man"):
        for mode in ("bg", "no_bg"):
            mask_images_from_json(
                json_path=json_paths[gender],
                source_folder=input_folders[gender],
                destination_folder=output_base / f"{gender}_{mode}",
                mode=mode,
                score_threshold=args.score_threshold,
                score_threshold_margin=args.score_margin,
            )

    print(f"Finished. Outputs are under: {output_base}")


if __name__ == "__main__":
    main()
