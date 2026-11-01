# PixelSGA

## Pipeline

The evaluation consists of four steps:

1. **Pixelation:** Generate pixelated images at multiple resolutions, together with grayscale and binary variants.
2. **Embedding extraction:** Extract image embeddings using a pretrained OpenCLIP, SigLIP, or DINOv2 model.
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

Embedding extraction supports OpenCLIP, SigLIP, and DINOv2 through OpenCLIP and Hugging Face Transformers.

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

Choose the embedding model using `--embedding_model`.

**OpenCLIP**

```bash
python embeddings_PixelSGA.py \
  --embedding_model clip \
  --base_input_path "/path/to/pixelation_output" \
  --output_path "/path/to/pixelsga_embeddings_clip" \
  --resolutions 224 112 56 28 14 7 4 2 1
  --batch_size 1
```

The default OpenCLIP configuration uses the `convnext_base_w` architecture with the pretrained weights `laion2b_s13b_b82k_augreg`.

**SigLIP**

```bash
python embeddings_PixelSGA.py \
  --embedding_model siglip \
  --base_input_path "/path/to/pixelation_output" \
  --output_path "/path/to/pixelsga_embeddings_siglip" \
  --resolutions 224 112 56 28 14 7 4 2 1
  --batch_size 1
```

The default model is `google/siglip-base-patch16-224`. Image embeddings are extracted using the model's pooled visual features.

**DINOv2 with dense patch features**

```bash
python embeddings_PixelSGA.py \
  --embedding_model dinov2 \
  --base_input_path "/path/to/pixelation_output" \
  --output_path "/path/to/pixelsga_embeddings_dinov2" \
  --resolutions 224 112 56 28 14 7 4 2 1 \
  --batch_size 1
```

The default model is `facebook/dinov2-with-registers-large`. The CLS token and register tokens are excluded, and the remaining patch-token features are averaged to produce one global descriptor per image. This is a dense-feature-based representation, rather than an embedding taken from the CLS token.

For all models, `--model_name` can be used to select another compatible model. The `--pretrained` option specifies OpenCLIP pretrained weights. 

***Batch size and batch-dependent artifacts***

The batch size must be set to `1` for all embedding extraction runs. Processing multiple images together may introduce batch-dependent artifacts into the extracted representations. In preliminary experiments, these artifacts were particularly problematic for highly pixelated images, such as 1×1 black-and-white images, where embeddings could become spuriously discriminative when images from different gender groups were processed in separate batches. This could inflate classification accuracy for reasons unrelated to the image content being evaluated.

Processing one image at a time avoids grouping images into batches by gender and ensures that each image is encoded independently. For this reason, use `--batch_size 1` for all models and experiments.


### 3. Run k-NN classification

```bash
python knn_classification_pixelSGA.py \
  --embeddings_dir "/path/to/pixelsga_embeddings_dinov2" \
  --output_json "/path/to/pixelsga_results_dinov2.json" \
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
  --json_path "/path/to/pixelsga_results_dinov2.json" \
  --output_csv_path "/path/to/pixelsga_summary_dinov2.csv" \
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

For reproducibility, record the embedding model, pretrained weights, and random seed used for each experiment. Different models and feature-extraction methods may produce different PixelSGA scores.

## Extra

We also provide the file 
```text
PixelSGA/
├── finetune_resnet.py
```
should the readers be interested in implementing the measure of PixelSGA via finetuning a ResNet model and performing balanced accuracy between the two classes under the same pixelation conditions explained before. 

```bash
python finetune_resnet.py \
  --male dir "/path/to/male_folder" \
  --female_dir "/path/to/female_folder" \
  --class-weighting loss
```


