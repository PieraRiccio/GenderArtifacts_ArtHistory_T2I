# PixelSGA

This folder contains the pipeline to compute the **PixelSGA** metric, which measures gender artifacts in images through pixel-space manipulations.

---

## Pipeline Overview

```
Raw Images
    │
    ▼
1. Image Processing       pixelation.py
    │
    ▼
2. CLIP Embedding         embeddings_pixelSGA.py
    │
    ▼
3. kNN Classification     knn_classification_pixelSGA.py
    │
    ▼
4. Metric Computation     pixelsga_computation.py
```

---

## Step-by-Step Instructions

### Step 1 — Image Processing
**Script:** `pixelation.py`

Process the raw images by applying pixelation at multiple resolutions and converting to grayscale and binary color spaces. This generates the full set of manipulated image variants used in the pipeline.

---

### Step 2 — CLIP Embedding Extraction
**Script:** `embeddings_pixelSGA.py`

Extract CLIP embeddings for each manipulated image folder produced in Step 1. Embeddings are saved as `.npy` files, one per image subfolder.

---

### Step 3 — kNN Classification
**Script:** `knn_classification_pixelSGA.py`

Run a leave-one-out balanced kNN classifier on the extracted embeddings. Balanced accuracy is computed for each resolution and color variant, and results are saved into a JSON file.

---

### Step 4 — Metric Computation
**Script:** `pixelsga_computation.py`

Aggregate the classification results from Step 3 to compute the final **PixelSGA** score. For each resolution, the best-performing color variant is selected; scores are then averaged across resolutions. Results are saved to a CSV file.
