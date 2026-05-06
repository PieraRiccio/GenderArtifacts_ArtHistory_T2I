import os
import json
import csv
from typing import Dict, Optional



def compute_sga2_from_dict(data: Dict) -> float:

    category = next(iter(data))
    results = data[category]

    bg = results["bg"]["balanced_accuracy"]
    no_bg = results["no_bg"]["balanced_accuracy"]

    return (bg + no_bg) / 2


def compute_sga2_from_folder(
    folder_path: str,
    output_csv_path: str,
    filename_key: str,
    skip_pattern: Optional[str] = None,
    sort_descending: bool = False,
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

        value = compute_sga2_from_dict(data)
        style = filename.split(filename_key)[1].replace(".json", "")
        print(f"{style}: {value:.4f}")

        results_list.append([style, value])

    if sort_descending:
        results_list.sort(key=lambda x: x[1], reverse=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["style", "sga2"])
        writer.writerows(results_list)

    print(f"\nResults saved to: {output_csv_path}")



if __name__ == "__main__":

    FOLDER_PATH      = "add your path here"   # folder with JSON files
    OUTPUT_CSV_PATH  = "add your path here"   # full path for output CSV
    FILENAME_KEY     = "k10_"                 # key to filter files and extract style name
    SKIP_PATTERN     = None                   # skip files containing this (None to disable)
    SORT_DESCENDING  = True

    compute_sga2_from_folder(
        folder_path=FOLDER_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        filename_key=FILENAME_KEY,
        skip_pattern=SKIP_PATTERN,
        sort_descending=SORT_DESCENDING,
    )
