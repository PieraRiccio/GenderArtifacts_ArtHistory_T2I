import os
import json
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import balanced_accuracy_score


def save_style_json(style, output_path):

    resolutions = ["224", "112", "56", "28", "14", "7", "4", "2", "1"]
    variants = ["", "g", "bw"]  # RGB, grayscale, binary

    data = {
        style: {
            f"{r}{v}": {} for r in resolutions for v in variants
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"JSON saved to:\n{output_path}")


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
    style,
    resolutions=None,
    k=5,
    random_state=13
):

    if resolutions is None:
        resolutions = ["224", "112", "56", "28", "14", "7", "4", "2", "1"]

    rng = np.random.default_rng(random_state)

    with open(json_path, "r") as f:
        data = json.load(f)

    files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith(".npy")])
    print(f"Found {len(files)} .npy files")

    # (filename suffix, json key suffix)
    variants = [
        ("", ""),         # RGB
        ("_gray", "g"),   # grayscale
        ("_bw", "bw"),    # binary B&W
    ]

    for r in resolutions:
        for file_suffix, key_suffix in variants:
            key = r + key_suffix
            print(f"\n=== Processing {key} ===")

            # Match exactly 2 files (female + male)
            matched = []
            for fname in files:
                if file_suffix:
                    if fname.endswith(f"_{r}{file_suffix}.npy"):
                        matched.append(fname)
                else:
                    if fname.endswith(f"_{r}.npy") and "_gray" not in fname and "_bw" not in fname:
                        matched.append(fname)

            print(f"Matched files: {len(matched)}")
            for fname in matched:
                print(f"  {fname}")

            if len(matched) != 2:
                print("❌ Expected exactly 2 files (female + male), skipping")
                continue

            X_list, y_list = [], []

            for fname in matched:
                emb = np.load(os.path.join(embeddings_dir, fname))
                print(f"  Loaded {fname} with shape {emb.shape}")

                if "female" in fname:
                    X_list.append(emb)
                    y_list.extend([1] * emb.shape[0])
                elif "male" in fname:
                    X_list.append(emb)
                    y_list.extend([0] * emb.shape[0])
                else:
                    raise ValueError(f"Gender not found in filename: {fname}")

            X = np.concatenate(X_list, axis=0)
            y = np.array(y_list)

            # Shuffle
            perm = rng.permutation(len(X))
            X, y = X[perm], y[perm]

            print(f"  Final X shape: {X.shape}")
            print(f"  Female samples: {(y == 1).sum()}")
            print(f"  Male samples:   {(y == 0).sum()}")

            # Run kNN and compute balanced accuracy
            y_pred = knn_leave_one_out_balanced_predictions(X, y, k=k)
            bal_acc = balanced_accuracy_score(y, y_pred)
            print(f"  ✅ Balanced accuracy: {bal_acc:.4f}")

            data[style][key]["balanced_accuracy"] = float(bal_acc)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ JSON written to disk:\n{json_path}")



if __name__ == "__main__":

    STYLE         = "add your style here"      # e.g. "NEUTRAL"
    EMBEDDINGS_DIR = "add your path here"      # folder with .npy files
    JSON_PATH      = "add your path here"      # path to output .json file
    K              = 5
    RESOLUTIONS    = ["224", "112", "56", "28", "14", "7", "4", "2", "1"]

    # Step 1: Create the JSON structure
    save_style_json(
        style=STYLE,
        output_path=JSON_PATH
    )

    # Step 2: Run kNN and populate the JSON
    run_knn_and_update_json(
        embeddings_dir=EMBEDDINGS_DIR,
        json_path=JSON_PATH,
        style=STYLE,
        resolutions=RESOLUTIONS,
        k=K
    )