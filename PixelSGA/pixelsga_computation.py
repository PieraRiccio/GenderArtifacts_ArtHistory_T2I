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
        candidates = [
            results[f"{res}{suffix}"]["balanced_accuracy"]
            for suffix in ["", "g", "bw"]
            if f"{res}{suffix}" in results
        ]
        if not candidates:
            raise ValueError(f"No values found for resolution {res}")
        selected_values.append(max(candidates))

    return sum(selected_values) / len(selected_values)


def compute_sga1_from_folder(
    folder_path: str,
    output_csv_path: str,
    filename_key: str,
    resolutions: Optional[List[str]] = None,
    skip_pattern: Optional[str] = None,
):

    results_list = []

    for filename in sorted(os.listdir(folder_path)):
        if not (filename.endswith(".json") and filename_key in filename):
            continue
        if skip_pattern and skip_pattern in filename:
            continue

        full_path = os.path.join(folder_path, filename)

        with open(full_path, "r") as f:
            data = json.load(f)

        value = compute_sga1_from_dict(data, resolutions=resolutions)
        style = filename.split(filename_key)[1].replace(".json", "")
        print(f"{style}: {value:.4f}")

        results_list.append([style, value])

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["style", "sga1"])
        writer.writerows(results_list)

    print(f"\nResults saved to: {output_csv_path}")


if __name__ == "__main__":

    FOLDER_PATH      = "add your path here"       # folder with JSON files
    OUTPUT_CSV_PATH  = "add your path here"       # full path for output CSV
    FILENAME_KEY     = "k5_"                      # key to filter files and extract style name
    SKIP_PATTERN     = "bg_k5"                    # skip files containing this (None to disable)
    RESOLUTIONS      = ["14", "7", "4", "2", "1"]

    compute_pixelsga_from_folder(
        folder_path=FOLDER_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        filename_key=FILENAME_KEY,
        resolutions=RESOLUTIONS,
        skip_pattern=SKIP_PATTERN,
    )
