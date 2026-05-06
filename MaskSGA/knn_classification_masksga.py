import os
import json
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import balanced_accuracy_score
 

 
def save_style_json(style, output_path, modes=("bg", "no_bg")):

    data = {style: {mode: {} for mode in modes}}
 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
 
    print(f"JSON saved to:\n{output_path}")
 
 
def knn_leave_one_out_balanced_predictions(X, y, k=5, random_state=42):

    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    y_pred = np.zeros(n, dtype=int)
 
    for i in range(n):
        X_ref = np.delete(X, i, axis=0)
        y_ref = np.delete(y, i)
 
        male_idx = np.where(y_ref == 0)[0]
        female_idx = np.where(y_ref == 1)[0]
 
        if len(male_idx) < len(female_idx):
            female_idx = rng.choice(female_idx, size=len(male_idx), replace=False)
        else:
            male_idx = rng.choice(male_idx, size=len(female_idx), replace=False)
 
        balanced_idx = np.concatenate([male_idx, female_idx])
        X_bal = X_ref[balanced_idx]
        y_bal = y_ref[balanced_idx]
 
        dists = cosine_distances(X[i].reshape(1, -1), X_bal).ravel()
        nn_idx = np.argsort(dists)[:k]
        y_pred[i] = Counter(y_bal[nn_idx]).most_common(1)[0][0]
 
    return y_pred
 
 
def run_knn_and_update_json(
    embeddings_dir,
    json_path,
    style,
    female_path_fn,
    male_path_fn,
    modes=("bg", "no_bg"),
    k=5,
    random_state=13
):

    rng = np.random.default_rng(random_state)
 
    print(f"Processing style: {style}")
 
    with open(json_path, "r") as f:
        data = json.load(f)
 
    for mode in modes:
        print(f"Processing mode: {mode}")
 
        feature_path_female = female_path_fn(embeddings_dir, style, mode)
        feature_path_male = male_path_fn(embeddings_dir, style, mode)
 
        emb_female = np.load(feature_path_female)
        emb_male = np.load(feature_path_male)
 
        X = np.concatenate([emb_female, emb_male], axis=0)
        y = np.array([1] * emb_female.shape[0] + [0] * emb_male.shape[0])
 
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]
 
        print(f"  Final X shape: {X.shape}")
        print(f"  Female samples: {(y == 1).sum()}")
        print(f"  Male samples:   {(y == 0).sum()}")
 
        y_pred = knn_leave_one_out_balanced_predictions(X, y, k=k)
        bal_acc = balanced_accuracy_score(y, y_pred)
        print(f"  ✅ Balanced accuracy: {bal_acc:.4f}")
 
        data[style][mode]["balanced_accuracy"] = float(bal_acc)
 
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
 
    print(f"\n✅ JSON written to disk:\n{json_path}")
 

 
if __name__ == "__main__":
 
    EMBEDDINGS_DIR = "add your path here"          # root embeddings folder
    OUTPUT_DIR     = "add your path here"          # folder for JSON output files
    STYLES         = ["add your styles here"]      # e.g. ["impressionism", "realism"]
    MODES          = ["bg", "no_bg"]
    K              = 5
 
    def female_path_fn(embeddings_dir, style, mode):
        return os.path.join(embeddings_dir, style, f"{style}_{mode}_female.npy")
 
    def male_path_fn(embeddings_dir, style, mode):
        return os.path.join(embeddings_dir, style, f"{style}_{mode}_male.npy")
 
    # Step 1: Create one JSON per style
    for style in STYLES:
        save_style_json(
            style=style,
            output_path=os.path.join(OUTPUT_DIR, f"bg_k{K}_{style}.json"),
            modes=MODES
        )
 
    # Step 2: Run kNN and populate each JSON
    for style in STYLES:
        run_knn_and_update_json(
            embeddings_dir=EMBEDDINGS_DIR,
            json_path=os.path.join(OUTPUT_DIR, f"bg_k{K}_{style}.json"),
            style=style,
            female_path_fn=female_path_fn,
            male_path_fn=male_path_fn,
            modes=MODES,
            k=K
        )