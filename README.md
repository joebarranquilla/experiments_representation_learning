# experiments_representation_learning

Benchmarking techniques on representation learning. This repository explores classical and modern approaches to learning meaningful data representations.

## Overview

Representation learning aims to discover useful structure in data through unsupervised or self-supervised learning. This project implements and compares several key techniques across different domains.

## Setup

Create and activate a conda environment named `rl-experiments`:

```bash
conda create -n rl-experiments python=3.11 -y
conda activate rl-experiments
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

To deactivate the environment when you're done:


## Techniques

### Principal Component Analysis (PCA)

Classical linear dimensionality reduction for CIFAR-10 with visualization and classification on PCA-compressed features.

- **pca-vanilla.py**: PCA benchmarking and reconstruction timing on flattened images.
- **pca-scratch.py**: Custom PCA implementation with t-SNE visualization and ground-truth labels.
- **pca_mlp_classifier.py**: Train a simple MLP or ConvNet for classification (supports PCA-based features or raw images).

### Autoencoders

Convolutional autoencoders learn compact image representations and can be used for linear probing (freeze encoder, train an MLP head for classification).

- **autoencoder.py**: Standard convolutional AE trained on CIFAR-10; includes linear probing pipeline.
- **vae.py**: Variational autoencoder with β-annealing for stable training; uses encoder `mu` for linear probing.

### Contrastive Learning (SimCLR)

Self-supervised contrastive pre-training with NT-Xent loss. Augmented views of the same image are pulled together while different images are pushed apart — no labels used.

- **simclr.py**: Train SimCLR on CIFAR-10 (ResNet-18, CIFAR-adapted stem), then freeze the encoder and train a linear probe. Two independent phases: `--pretrain` and `--probe`.

## Project Structure

```
├── README.md
├── requirements.txt
├── pca-vanilla.py              # PCA benchmarking
├── pca-scratch.py              # Custom PCA + t-SNE visualization
├── pca_mlp_classifier.py       # MLP/ConvNet classifier with PCA features
├── autoencoder.py              # Convolutional AE + linear probe
├── vae.py                      # VAE + linear probe
└── simclr.py                   # SimCLR contrastive pre-training + linear probe
```

## Usage

### PCA Baselines

```bash
# Vanilla PCA: benchmark components, reconstruction error
python pca-vanilla.py

# Custom PCA: visualize with t-SNE and ground-truth labels
python pca-scratch.py

# Train classifier on PCA features (or raw images with ConvNet)
python pca_mlp_classifier.py    # default: MLP on PCA
model_type='conv' python pca_mlp_classifier.py  # ConvNet on images
```

### Autoencoders

```bash
# Train AE, save encoder, then run linear probe
python autoencoder.py

# Skip training, load checkpoint and re-probe
python autoencoder.py --probe-only

# Train VAE with β-annealing, save encoder, run linear probe
python vae.py

# Load VAE checkpoint and re-probe
python vae.py --probe-only
```

### SimCLR

```bash
# Phase 1: contrastive pre-training (~60-90 min on CPU for 30 epochs)
python simclr.py --pretrain

# Phase 2: linear probing on the frozen encoder (auto-loads latest checkpoint)
python simclr.py --probe

# Both phases back-to-back
python simclr.py --pretrain --probe

# Fewer epochs for a quick test
python simclr.py --pretrain --pretrain-epochs 10 --probe

# Load a specific checkpoint (e.g. from sthalles/SimCLR)
python simclr.py --probe --checkpoint path/to/checkpoint_0100.pth.tar
```

> **Note**: SimCLR must be trained on CIFAR-10 (no public PyTorch pretrained checkpoints exist for this dataset). The checkpoint format is compatible with [sthalles/SimCLR](https://github.com/sthalles/SimCLR).

All scripts save checkpoints to `checkpoints/` and generate visualizations (reconstructions, predictions) as PNG files.

All scripts run on CPU; CUDA will be used if available.
