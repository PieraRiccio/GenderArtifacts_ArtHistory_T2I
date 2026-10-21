import argparse
import json
import os
from collections import Counter

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_distances


def knn_leave_one_out_balanced_predictions(X, y, k=5, random_state=42):
    """Predict each sample using leave-one-out kNN with balanced reference classes."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    y_pred = np.zeros(n, dtype=int)

    for i in range(n):
        X_ref = np.delete(X, i, axis=0)
        y_ref = np.delete(y, i)

        male_idx = np.where(y_ref == 0)[0]
        female_idx = np.where(y_ref == 1)[0]

        if len(male_idx) == 0 or len(female_idx) == 0:
            raise ValueError(
                "Each class must have at least one reference sample in every "
                "leave-one-out iteration. Provide at least two samples per gender."
            )

        if len(male_idx) < len(female_idx):
            female_idx = rng.choice(female_idx, size=len(male_idx), replace=False)
        elif len(male_idx) > len(female_idx):
            male_idx = rng.choice(male_idx, size=len(female_idx), replace=False)

        balanced_idx = np.concatenate([male_idx, female_idx])
        X_bal = X_ref[balanced_idx]
        y_bal = y_ref[balanced_idx]

        effective_k = min(k, len(y_bal))
        dists = cosine_distances(X[i].reshape(1, -1), X_bal).ravel()
        nn_idx = np.argsort(dists)[:effective_k]
        y_pred[i] = Counter(y_bal[nn_idx]).most_common(1)[0][0]

    return y_pred


def run_knn(
    embeddings_dir,
    output_json,
    modes=("bg", "no_bg"),
    k=5,
    random_state=13,
):
    """Run balanced leave-one-out kNN for both masking modes and save results."""
    results = {}

    for mode in modes:
        female_path = os.path.join(embeddings_dir, f"{mode}_female.npy")
        male_path = os.path.join(embeddings_dir, f"{mode}_male.npy")

        for path in (female_path, male_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Embedding file not found: {path}")

        emb_female = np.load(female_path)
        emb_male = np.load(male_path)

        if emb_female.ndim != 2 or emb_male.ndim != 2:
            raise ValueError(f"Expected 2D embedding arrays for mode '{mode}'.")
        if emb_female.shape[1] != emb_male.shape[1]:
            raise ValueError(
                f"Embedding dimensions differ for mode '{mode}': "
                f"female={emb_female.shape[1]}, male={emb_male.shape[1]}"
            )
        if len(emb_female) < 2 or len(emb_male) < 2:
            raise ValueError(
                f"Mode '{mode}' needs at least two female and two male samples "
                "for leave-one-out balanced kNN."
            )

        X = np.concatenate([emb_female, emb_male], axis=0)
        y = np.array([1] * len(emb_female) + [0] * len(emb_male))

        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]

        print(f"\nProcessing mode: {mode}")
        print(f"  Embeddings shape: {X.shape}")
        print(f"  Female samples: {(y == 1).sum()}")
        print(f"  Male samples:   {(y == 0).sum()}")

        y_pred = knn_leave_one_out_balanced_predictions(
            X, y, k=k, random_state=random_state
        )
        bal_acc = balanced_accuracy_score(y, y_pred)
        results[mode] = {"balanced_accuracy": float(bal_acc)}
        print(f"  Balanced accuracy: {bal_acc:.4f}")

    output_parent = os.path.dirname(os.path.abspath(output_json))
    os.makedirs(output_parent, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults JSON saved to:\n{output_json}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run balanced leave-one-out kNN on MaskSGA embeddings for bg and no_bg."
        )
    )
    parser.add_argument(
        "--embeddings_dir",
        required=True,
        help="Folder containing bg_female.npy, bg_male.npy, no_bg_female.npy, no_bg_male.npy.",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Path where the results JSON will be saved.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors (default: 5).")
    parser.add_argument(
        "--random_state", type=int, default=13, help="Random seed (default: 13)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1.")

    run_knn(
        embeddings_dir=args.embeddings_dir,
        output_json=args.output_json,
        k=args.k,
        random_state=args.random_state,
    )
