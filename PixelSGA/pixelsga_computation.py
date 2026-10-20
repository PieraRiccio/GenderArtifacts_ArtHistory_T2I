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

    # Support both flat and older nested JSON structures.
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


def compute_sga1_from_file(
    json_path: str,
    output_csv_path: str,
    resolutions: Optional[List[str]] = None,
):

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    sga1 = compute_sga1_from_dict(data, resolutions=resolutions)

    case = os.path.splitext(os.path.basename(json_path))[0]
    print(f"{case}: {sga1:.4f}")

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["case", "sga1"])
        writer.writerow([case, sga1])

    print(f"Results saved to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute SGA1 from one KNN result JSON file."
    )

    parser.add_argument(
        "--json_path",
        required=True,
        help="Path to the KNN result JSON file."
    )
    parser.add_argument(
        "--output_csv_path",
        required=True,
        help="Path for the output CSV file."
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["14", "7", "4", "2", "1"],
        help="Resolutions to include in the SGA1 calculation."
    )

    args = parser.parse_args()

    compute_sga1_from_file(
        json_path=args.json_path,
        output_csv_path=args.output_csv_path,
        resolutions=args.resolutions,
    )
