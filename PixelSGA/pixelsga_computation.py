
import os
import json
import csv
import argparse
from typing import Dict, List, Optional


def compute_sga1_from_dict(
    data: Dict,
    resolutions: Optional[List[str]] = None,
) -> float:

    if resolutions is None:
        resolutions = ["14", "7", "4", "2", "1"]

    # Current KNN output has resolution keys directly at the top level.
    # Also support the older nested JSON structure.
    results = data
    if not any(
        f"{res}{suffix}" in results
        for res in resolutions
        for suffix in ["", "g", "bw"]
    ):
        if data and isinstance(next(iter(data.values())), dict):
            results = next(iter(data.values()))

    selected_values = []

    for res in resolutions:
        candidates = [
            results[key]["balanced_accuracy"]
            for suffix in ["", "g", "bw"]
            if (key := f"{res}{suffix}") in results
            and isinstance(results[key], dict)
            and "balanced_accuracy" in results[key]
        ]

        if not candidates:
            raise ValueError(
                f"No balanced accuracy values found for resolution {res}"
            )

        selected_values.append(max(candidates))

    return sum(selected_values) / len(selected_values)


def compute_sga1_from_folder(
    folder_path: str,
    output_csv_path: str,
    filename_key: str = "k5_",
    resolutions: Optional[List[str]] = None,
    skip_pattern: Optional[str] = "bg_k5",
):

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Input folder not found: {folder_path}")

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

        # Use the filename to identify each case.
        case = filename.split(filename_key, 1)[1].removesuffix(".json")

        print(f"{case}: {value:.4f}")
        results_list.append([case, value])

    if not results_list:
        raise ValueError(
            f"No matching JSON files found in {folder_path}. "
            f"Check --filename_key and --skip_pattern."
        )

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["case", "sga1"])
        writer.writerows(results_list)

    print(f"\nResults saved to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute SGA1 scores from KNN result JSON files."
    )

    parser.add_argument("--folder_path", required=True,
                        help="Folder containing the result JSON files.")
    parser.add_argument("--output_csv_path", required=True,
                        help="Path for the output CSV file.")
    parser.add_argument("--filename_key", default="k5_",
                        help="Only process filenames containing this string.")
    parser.add_argument("--skip_pattern", default="bg_k5",
                        help="Skip filenames containing this string. "
                             "Pass an empty string to disable.")
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["14", "7", "4", "2", "1"],
        help="Resolutions to include in the SGA1 calculation."
    )

    args = parser.parse_args()

    compute_sga1_from_folder(
        folder_path=args.folder_path,
        output_csv_path=args.output_csv_path,
        filename_key=args.filename_key,
        resolutions=args.resolutions,
        skip_pattern=args.skip_pattern or None,
    )
