# PixelSGA

## Pipeline

The evaluation consists of four steps:

1. **Pixelation:** Generate pixelated images at multiple resolutions, together with grayscale and binary variants.
2. **Embedding extraction:** Extract image embeddings using a pretrained OpenCLIP or SigLIP model.
3. **k-NN classification:** Evaluate the embeddings using balanced leave-one-out predictions and compute balanced accuracy.
4. **PixelSGA computation:** Aggregate the balanced accuracy results into a single PixelSGA score.

Each script can be executed independently through command-line arguments.

## Repository structure

```text
PixelSGA/
├── pixelation.py
├── embeddings_PixelSGA.py
├── knn_classification_pixelSGA.py
├── pixelsga_computation.py
├── requirements.txt
└── README.md
```

## Requirements

Install the dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

Embedding extraction supports OpenCLIP and SigLIP. The required packages for both models are included in `requirements.txt`.

## Usage

The examples below illustrate how to run the pipeline for one case. Replace the example paths with the paths to your data.

### 1. Pixelate images

```bash
python pixelation.py \
  --input_female "/path/to/female_images" \
  --input_male "/path/to/male_images" \
  --output_base "/path/to/pixelation_output"
```

The script generates processed image collections at the configured resolutions:

`224, 112, 56, 28, 14, 7, 4, 2, 1`

It also generates grayscale and binary variants.

### 2. Extract embeddings

Select the embedding model using `--embedding_model`.

**OpenCLIP:**

```bash
python embeddings_PixelSGA.py \
  --embedding_model clip \
  --base_input_path "/path/to/pixelation_output" \
  --output_path "/path/to/pixelsga_embeddings_clip" \
  --resolutions 224 112 56 28 14 7 4 2 1
```

**SigLIP:**

```bash
python embeddings_PixelSGA.py \
  --embedding_model siglip \
  --base_input_path "/path/to/pixelation_output" \
  --output_path "/path/to/pixelsga_embeddings_siglip" \
  --resolutions 224 112 56 28 14 7 4 2 1
```

The default SigLIP model is `google/siglip-base-patch16-224`. The OpenCLIP model uses the architecture and pretrained weights configured in the script. Optional model settings can be specified using `--model_name` and `--pretrained`, where supported.

Use separate output directories for different embedding models to avoid overwriting their embeddings.

### 3. Run k-NN classification

```bash
python knn_classification_pixelSGA.py \
  --embeddings_dir "/path/to/pixelsga_embeddings_siglip" \
  --output_json "/path/to/pixelsga_results_siglip.json" \
  --k 5 \
  --resolutions 224 112 56 28 14 7 4 2 1 \
  --random_state 13
```

The script evaluates the embeddings and saves balanced accuracy results to a JSON file.

The JSON keys identify the resolution and image variant:

- `"224"`: original pixelated images at resolution 224
- `"224g"`: grayscale images at resolution 224
- `"224bw"`: binary images at resolution 224

The same naming convention is used for the other resolutions.

### 4. Compute PixelSGA

```bash
python pixelsga_computation.py \
  --json_path "/path/to/pixelsga_results_siglip.json" \
  --output_csv_path "/path/to/pixelsga_summary_siglip.csv" \
  --resolutions 14 7 4 2 1
```

The script computes the PixelSGA score from a single JSON result file and saves it to a CSV file.

For each selected resolution, it takes the maximum balanced accuracy among the available original, grayscale, and binary variants, then averages these maxima. By default, PixelSGA uses resolutions `14, 7, 4, 2, 1`.

## Outputs

The pipeline produces:

- **Processed images:** Pixelated, grayscale, and binary image collections.
- **Embeddings:** NumPy (`.npy`) files containing image representations.
- **Evaluation results:** A JSON file containing balanced accuracy values.
- **PixelSGA summary:** A CSV file containing the computed PixelSGA score for the case.

Each case and embedding model should have its own output paths to avoid overwriting results from previous runs.

For reproducibility, record the embedding model, pretrained weights, and random seed used for each experiment. Different model configurations may produce different PixelSGA scores.
