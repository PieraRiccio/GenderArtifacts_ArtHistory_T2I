import os
import json
import csv
from typing import Dict, List, Optional


def compute_sga1_from_dict(data: Dict, resolutions: Optional[List[str]] = None) -> float:

    if resolutions is None:
        resolutions = ["14", "7", "4", "2", "1"]

    category = next(iter(data))
    results = data[category]

    selected_values = []

    for res in resolutions:
        candidates = []
        for suffix in ["", "g", "bw"]:
            key = f"{res}{suffix}"
            if key in results:
                candidates.append(results[key]["balanced_accuracy"])

        if not candidates:
            raise ValueError(f"No values found for resolution {res}")

        selected_values.append(max(candidates))

    return sum(selected_values) / len(selected_values)


def compute_sga1_from_folder(
    folder_path: str,
    output_csv_path: str,
    resolutions: Optional[List[str]] = None,
    filename_key: str = "k5_",
    skip_pattern: Optional[str] = None,
):

    results_list = []

    for filename in os.listdir(folder_path):
        if not (filename.endswith(".json") and filename_key in filename):
            continue

        if skip_pattern and skip_pattern in filename:
            continue

        full_path = os.path.join(folder_path, filename)

        with open(full_path, "r") as f:
            data = json.load(f)

        sga1_value = compute_sga1_from_dict(data, resolutions=resolutions)

        # Extract style name: text after filename_key and before ".json"
        style = filename.split(filename_key)[1].replace(".json", "")
        print(f"{style}: {sga1_value:.4f}")

        results_list.append([style, sga1_value])

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["style", "sga1"])
        writer.writerows(results_list)

    print(f"\nResults saved to: {output_csv_path}")


if __name__ == "__main__":

    FOLDER_PATH      = "add your path here"       # folder with JSON files
    OUTPUT_CSV_PATH  = "add your path here"       # full path for output CSV
    RESOLUTIONS      = ["14", "7", "4", "2", "1"] # resolutions for PixelSGA
    FILENAME_KEY     = "k5_"                      # key to find/extract style name
    SKIP_PATTERN     = "bg_k5"                    # filenames containing this are skipped

    compute_sga1_from_folder(
        folder_path=FOLDER_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        resolutions=RESOLUTIONS,
        filename_key=FILENAME_KEY,
        skip_pattern=SKIP_PATTERN,
    )