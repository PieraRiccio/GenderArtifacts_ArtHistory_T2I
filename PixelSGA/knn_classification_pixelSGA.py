import os
import json
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import balanced_accuracy_score
import argparse


def save_results_json(output_path, resolutions):
    """Initialize a JSON file without requiring a style label."""
    variants = ["", "g", "bw"]

    data = {
        f"{r}{v}": {}
        for r in map(str, resolutions)
        for v in variants
    }

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"JSON initialized:\n{output_path}")


def knn_leave_one_out_balanced_predictions(X, y, k=5, random_state=42):

    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    y_pred = np.zeros(n, dtype=int)

    for i in range(n):
        # Remove query image
        X_ref = np.delete(X, i, axis=0)
        y_ref = np.delete(y, i)

        # Split by class
        male_idx = np.where(y_ref == 0)[0]
        female_idx = np.where(y_ref == 1)[0]

        # Subsample majority class
        if len(male_idx) < len(female_idx):
            female_idx = rng.choice(female_idx, size=len(male_idx), replace=False)
        else:
            male_idx = rng.choice(male_idx, size=len(female_idx), replace=False)

        balanced_idx = np.concatenate([male_idx, female_idx])
        X_bal = X_ref[balanced_idx]
        y_bal = y_ref[balanced_idx]

        # Cosine distances and kNN vote
        dists = cosine_distances(X[i].reshape(1, -1), X_bal).ravel()
        nn_idx = np.argsort(dists)[:k]
        nn_labels = y_bal[nn_idx]

        y_pred[i] = Counter(nn_labels).most_common(1)[0][0]

    return y_pred

def run_knn_and_update_json(
    embeddings_dir,
    json_path,
    resolutions=None,
    k=5,
    random_state=13,
):
    """Evaluate male/female embeddings and save results by resolution."""

    if resolutions is None:
        resolutions = ["224", "112", "56", "28", "14", "7", "4", "2", "1"]

    resolutions = [str(r) for r in resolutions]
    rng = np.random.default_rng(random_state)

    with open(json_path, "r") as f:
        data = json.load(f)

    files = sorted(
        f for f in os.listdir(embeddings_dir)
        if f.endswith(".npy")
    )
    print(f"Found {len(files)} .npy files")

    variants = [
        ("", ""),
        ("_gray", "g"),
        ("_bw", "bw"),
    ]

    for r in resolutions:
        for file_suffix, key_suffix in variants:
            key = r + key_suffix
            print(f"\n=== Processing {key} ===")

            matched = []
            for fname in files:
                if file_suffix:
                    if fname.endswith(f"_{r}{file_suffix}.npy"):
                        matched.append(fname)
                else:
                    if (
                        fname.endswith(f"_{r}.npy")
                        and "_gray" not in fname
                        and "_bw" not in fname
                    ):
                        matched.append(fname)

            print(f"Matched files: {len(matched)}")
            for fname in matched:
                print(f"  {fname}")

            if len(matched) != 2:
                print("Expected exactly two files (male + female); skipping.")
                continue

            X_list, y_list = [], []
            found_labels = set()

            for fname in matched:
                emb = np.load(os.path.join(embeddings_dir, fname))

                if emb.ndim != 2:
                    raise ValueError(
                        f"{fname} must contain a 2D array; got {emb.shape}"
                    )

                print(f"  Loaded {fname}: {emb.shape}")

                lower_name = fname.lower()
                if "female" in lower_name:
                    label = 1
                elif "male" in lower_name:
                    label = 0
                else:
                    raise ValueError(
                        f"Cannot determine gender from filename: {fname}"
                    )

                if label in found_labels:
                    raise ValueError(
                        f"Duplicate class found among matched files: {fname}"
                    )

                found_labels.add(label)
                X_list.append(emb)
                y_list.extend([label] * len(emb))

            if len({x.shape[1] for x in X_list}) != 1:
                raise ValueError(
                    f"Embedding dimensions differ for resolution {r}"
                )

            X = np.concatenate(X_list, axis=0)
            y = np.asarray(y_list)

            # Shuffle the combined dataset
            perm = rng.permutation(len(X))
            X, y = X[perm], y[perm]

            print(f"  Final X shape: {X.shape}")
            print(f"  Female samples: {(y == 1).sum()}")
            print(f"  Male samples:   {(y == 0).sum()}")

            y_pred = knn_leave_one_out_balanced_predictions(
                X, y, k=k, random_state=random_state
            )
            bal_acc = balanced_accuracy_score(y, y_pred)

            print(f"  Balanced accuracy: {bal_acc:.4f}")

            data.setdefault(key, {})
            data[key]["balanced_accuracy"] = float(bal_acc)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to:\n{json_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run balanced leave-one-out kNN on male/female embeddings."
    )
    parser.add_argument(
        "--embeddings_dir",
        required=True,
        help="Directory containing male and female .npy embeddings.",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Output JSON path for classification results.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--resolutions",
        type=str,
        nargs="+",
        default=["224", "112", "56", "28", "14", "7", "4", "2", "1"],
    )
    parser.add_argument("--random_state", type=int, default=13)

    args = parser.parse_args()

    save_results_json(args.output_json, args.resolutions)

    run_knn_and_update_json(
        embeddings_dir=args.embeddings_dir,
        json_path=args.output_json,
        resolutions=args.resolutions,
        k=args.k,
        random_state=args.random_state,
    )
