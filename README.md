# CNN-partial-face-identification
Partial face identification using CNNs and transfer learning.

## Overview

Face recognition systems usually perform well when faces are frontal, clear, and fully visible. However, their performance decreases when faces are partially occluded by glasses, masks, scarves, hats, poor lighting, or incomplete camera framing.

This project focuses on partial face identification, where the goal is to identify individuals from incomplete or occluded facial images. The proposed framework uses transfer learning with pretrained CNN model and additional trainable classification layers.

## Key Features

- Partial face identification under occlusion
- Transfer learning using pretrained CNN architectures
- Experiments with FaceNet
- Data augmentation using Keras ImageDataGenerator
- Evaluation using accuracy, precision, recall, and F1-score
- Reproducible training and evaluation pipeline

## Model

The repository includes experiments using:

| Model | Description |
|---|---|
| FaceNet / PFN | Final FaceNet-based model with added trainable layers |

## Dataset

This project uses the Specs on Faces (SoF) dataset, which contains facial images with glasses, occlusion, and illumination variations.

The dataset should be organized as follows:

```text
data/
├── train/
│   ├── person_001/
│   ├── person_002/
│   └── ...
├── val/
│   ├── person_001/
│   ├── person_002/
│   └── ...
└── test/
    ├── person_001/
    ├── person_002/
    └── ...
