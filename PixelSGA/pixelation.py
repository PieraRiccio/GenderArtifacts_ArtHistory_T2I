import os
from PIL import Image
import argparse



def pixelate_and_upscale_folder(
    input_folder,
    output_folder,
    pixel_size,
    target_size=(224, 224),
    downsample_resample=Image.NEAREST,
    upsample_resample=Image.NEAREST,
    extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")
):

    os.makedirs(output_folder, exist_ok=True)

    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith(extensions):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with Image.open(input_path) as img:
            img = img.convert("RGB")

            # 1) Downsample to low resolution
            low_res = img.resize(
                (pixel_size, pixel_size),
                resample=downsample_resample
            )

            # 2) Upsample back to target size
            upscaled = low_res.resize(
                target_size,
                resample=upsample_resample
            )

            upscaled.save(output_path)

    print(
        f"Saved pixelated images to {output_folder} "
        f"(effective resolution {pixel_size}×{pixel_size}, "
        f"final size {target_size[0]}×{target_size[1]})"
    )


def copy_folder_to_grayscale(
    input_folder,
    output_folder,
    extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")
):

    os.makedirs(output_folder, exist_ok=True)

    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(extensions):
            in_path = os.path.join(input_folder, filename)
            out_path = os.path.join(output_folder, filename)

            with Image.open(in_path) as img:
                gray = img.convert("L")
                gray.save(out_path)

    print(f"Grayscale images saved to:\n{output_folder}")


def copy_folder_to_binary(
    input_folder,
    output_folder,
    threshold=128,
    extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")
):

    os.makedirs(output_folder, exist_ok=True)

    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(extensions):
            in_path = os.path.join(input_folder, filename)
            out_path = os.path.join(output_folder, filename)

            with Image.open(in_path) as img:
                gray = img.convert("L")
                bw = gray.point(
                    lambda p: 255 if p >= threshold else 0,
                    mode="1"
                )
                bw.save(out_path)

    print(f"Binary (black & white) images saved to:\n{output_folder}")


def convert_folder_to_binary(folder_path, threshold=128):

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path)
                gray = img.convert("L")
                binary = gray.point(lambda x: 255 if x > threshold else 0, mode='1')
                binary.save(file_path)
                print(f"Converted: {filename}")
            except Exception as e:
                print(f"Failed on {filename}: {e}")

    print("Done!")



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--input_female", required=True)
    parser.add_argument("--input_male", required=True)
    parser.add_argument("--output_base", required=True)
    args = parser.parse_args()

    INPUT_FEMALE = args.input_female
    INPUT_MALE = args.input_male
    OUTPUT_BASE = args.output_base


    resolutions     = [224, 112, 56, 28, 14, 7, 4, 2, 1]
    resolutions_sub = [112, 56, 28, 14, 7, 4, 2, 1]


    for r in resolutions:
        pixelate_and_upscale_folder(
            input_folder=INPUT_FEMALE,
            output_folder=os.path.join(OUTPUT_BASE, f"subres_{r}", f"female_{r}"),
            pixel_size=r
        )
        pixelate_and_upscale_folder(
            input_folder=INPUT_MALE,
            output_folder=os.path.join(OUTPUT_BASE, f"subres_{r}", f"male_{r}"),
            pixel_size=r
        )

    copy_folder_to_grayscale(
        input_folder=os.path.join(OUTPUT_BASE, "subres_224", "female_224"),
        output_folder=os.path.join(OUTPUT_BASE, "subres_224", "female_224_gray")
    )
    copy_folder_to_grayscale(
        input_folder=os.path.join(OUTPUT_BASE, "subres_224", "male_224"),
        output_folder=os.path.join(OUTPUT_BASE, "subres_224", "male_224_gray")
    )

    copy_folder_to_binary(
        input_folder=os.path.join(OUTPUT_BASE, "subres_224", "female_224"),
        output_folder=os.path.join(OUTPUT_BASE, "subres_224", "female_224_bw")
    )
    copy_folder_to_binary(
        input_folder=os.path.join(OUTPUT_BASE, "subres_224", "male_224"),
        output_folder=os.path.join(OUTPUT_BASE, "subres_224", "male_224_bw")
    )

    for r in resolutions_sub:
        pixelate_and_upscale_folder(
            input_folder=os.path.join(OUTPUT_BASE, "subres_224", "female_224_gray"),
            output_folder=os.path.join(OUTPUT_BASE, f"subres_{r}", f"female_{r}_gray"),
            pixel_size=r
        )
        pixelate_and_upscale_folder(
            input_folder=os.path.join(OUTPUT_BASE, "subres_224", "male_224_gray"),
            output_folder=os.path.join(OUTPUT_BASE, f"subres_{r}", f"male_{r}_gray"),
            pixel_size=r
        )

    for r in resolutions_sub:
        pixelate_and_upscale_folder(
            input_folder=os.path.join(OUTPUT_BASE, "subres_224", "female_224_bw"),
            output_folder=os.path.join(OUTPUT_BASE, f"subres_{r}", f"female_{r}_bw"),
            pixel_size=r
        )
        pixelate_and_upscale_folder(
            input_folder=os.path.join(OUTPUT_BASE, "subres_224", "male_224_bw"),
            output_folder=os.path.join(OUTPUT_BASE, f"subres_{r}", f"male_{r}_bw"),
            pixel_size=r
        )

    for r in resolutions_sub:
        convert_folder_to_binary(
            os.path.join(OUTPUT_BASE, f"subres_{r}", f"female_{r}_bw")
        )
        convert_folder_to_binary(
            os.path.join(OUTPUT_BASE, f"subres_{r}", f"male_{r}_bw")
        )
