# MaskSGA

This folder contains the pipeline to compute the **MaskSGA** metric, which measures gender artifacts in images by isolating and comparing foreground (subject) and background regions through bounding box masking.

---

## Pipeline Overview

```
Raw Images
    │
    ▼
1. Mask Generation        generate_masked_images.py
    │
    ▼
2. CLIP Embedding         embeddings_maskSGA.py
    │
    ▼
3. kNN Classification     knn_classification_maskSGA.py
    │
    ▼
4. Metric Computation     masksga_computation.py
```

---

## Step-by-Step Instructions

### Step 1 — Mask Generation
**Script:** `generate_masked_images.py`

Compute bounding boxes around the subject in each image and generate two masked variants per image: one retaining the background (`bg`) and one retaining the foreground without background (`no_bg`).

---

### Step 2 — CLIP Embedding Extraction
**Script:** `embeddings_maskSGA.py`

Extract CLIP embeddings for each masked image folder produced in Step 1. Embeddings are saved as `.npy` files, one per (style, mode, gender) combination.

---

### Step 3 — kNN Classification
**Script:** `knn_classification_maskSGA.py`

Run a leave-one-out balanced kNN classifier on the extracted embeddings. Balanced accuracy is computed separately for the `bg` and `no_bg` conditions, and results are saved into a JSON file.

---

### Step 4 — Metric Computation
**Script:** `masksga_computation.py`

Aggregate the classification results from Step 3 to compute the final **MaskSGA** score, defined as the average balanced accuracy across the `bg` and `no_bg` conditions. Results are saved to a CSV file.
