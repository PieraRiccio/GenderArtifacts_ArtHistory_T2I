import argparse
import json
import os


def compute_sga2(json_path: str) -> float:
    """
    Compute MaskSGA as the average balanced accuracy
    across the bg and no_bg conditions.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    bg = data["bg"]["balanced_accuracy"]
    no_bg = data["no_bg"]["balanced_accuracy"]

    sga2 = (bg + no_bg) / 2

    print(f"BG balanced accuracy:    {bg:.4f}")
    print(f"No-BG balanced accuracy: {no_bg:.4f}")
    print(f"MaskSGA (average):       {sga2:.4f}")

    return sga2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute MaskSGA from a kNN classification results JSON."
    )
    parser.add_argument(
        "--json_path",
        required=True,
        help="Path to the JSON file containing bg and no_bg balanced accuracies.",
    )
    args = parser.parse_args()

    compute_sga2(args.json_path)
