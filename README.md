# Gender Artifacts from Art History to Text-to-Image Generation
Piera Riccio, Miriam Doh, Benedikt Hoeltgen, Noa Garcia, Nanne van Noord

This repository contains the code for **“Gender Artifacts from Art History to Text-to-Image Generation,”** a paper to be published at **NeurIPS 2026** Evaluation & Datasets track.

We provide two complementary metrics for investigating gender-related artifacts in sets of images, each implemented in its own directory.

## Metrics

| Metric | Description | Implementation |
|---|---|---|
| **PixelSGA** | Evaluates gender artifacts under pixel-space manipulations, including changes in image resolution and color. | [`PixelSGA/`](PixelSGA/) |
| **MaskSGA** | Evaluates gender artifacts using foreground and background masking to examine the role of people and their visual context. | [`MaskSGA/`](MaskSGA/) |

## Getting Started

Each directory contains the scripts, dependencies, and instructions required to compute its respective metric.

- **PixelSGA:** See [`PixelSGA/README.md`](PixelSGA/README.md) for setup instructions and the complete computation pipeline.
- **MaskSGA:** See [`MaskSGA/README.md`](MaskSGA/README.md) for setup instructions and the complete computation pipeline.

We recommend consulting the README in the relevant directory before running either pipeline.

## Repository Structure

```text
GenderArtifacts_ArtHistory_T2I/
├── PixelSGA/
│   └── README.md
├── MaskSGA/
│   └── README.md
└── README.md
```

Each metric is designed to be run independently. Please refer to its corresponding README for the required inputs, configuration options, and expected outputs.
