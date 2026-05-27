"""
Shared utilities for CIFAR-10 representation-learning experiments.

Centralised here to avoid repeating data-loading, probing, and visualisation
code across autoencoder.py, vae.py, and simclr.py.

Public API
----------
Constants      : CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD
Data           : get_cifar10_loaders
Probing        : LinearProbe, train_probe_epoch, eval_probe
Visualisation  : visualize_reconstructions, visualize_probe_predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────

def get_cifar10_loaders(
    data_root: str = "data",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    augment_train: bool = False,
    normalize: bool = False,
):
    """
    Return (train_loader, val_loader, test_loader) for CIFAR-10.

    augment_train : apply RandomHorizontalFlip + RandomCrop(32, padding=4) to
                    the training split only — the biggest single accuracy lever
                    for reconstruction-based encoder probing.
    normalize     : apply CIFAR-10 channel-wise mean/std normalisation.  Set
                    True when feeding a ResNet-style backbone (e.g. simclr.py).
                    Val and test always share the same transforms as train
                    (without augmentation).

    The train/val split uses a fixed seed (42) for reproducibility.  The two
    dataset objects created for train (aug) and val (clean) share the same
    random split indices so val always sees clean images.
    """
    base = [transforms.ToTensor()]
    if normalize:
        base.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))

    clean_transform = transforms.Compose(base)
    aug_transform   = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        *base,
    ])
    train_transform = aug_transform if augment_train else clean_transform

    train_full_aug = CIFAR10(root=data_root, train=True,  download=True,
                             transform=train_transform)
    train_full_val = CIFAR10(root=data_root, train=True,  download=True,
                             transform=clean_transform)
    test_set       = CIFAR10(root=data_root, train=False, download=True,
                             transform=clean_transform)

    n_val   = int(len(train_full_aug) * val_fraction)
    n_train = len(train_full_aug) - n_val
    indices = torch.randperm(
        len(train_full_aug),
        generator=torch.Generator().manual_seed(42),
    ).tolist()
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_loader = DataLoader(
        Subset(train_full_aug, train_idx), batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(train_full_val, val_idx), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    print(f"Train: {n_train} | Val: {n_val} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────
# Linear probe
# ──────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """
    MLP classification head on a *frozen* encoder.

    Parameters
    ----------
    encoder     : nn.Module  – model whose parameters are frozen.  Its
                  ``forward`` is used to extract features unless
                  ``encode_fn`` overrides it.
    in_dim      : int        – dimensionality of the encoder output.
    encode_fn   : callable   – optional override for feature extraction.
                  Pass ``vae.encode_mu`` for VAE or ``simclr.encode`` for
                  SimCLR to use the appropriate inference method.
                  If None, ``encoder(x)`` is called directly.
    hidden_dims : list[int]  – hidden layer sizes; empty ⟹ pure linear probe.
    num_classes : int        – output classes (10 for CIFAR-10).
    """

    def __init__(
        self,
        encoder: nn.Module,
        in_dim: int,
        encode_fn=None,
        hidden_dims: list = None,
        num_classes: int = 10,
    ):
        super().__init__()
        self.encoder = encoder
        self._encode = encode_fn if encode_fn is not None else encoder

        for p in self.encoder.parameters():
            p.requires_grad = False

        layers, d = [], in_dim
        for h in (hidden_dims or []):
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self._encode(x)
        return self.head(z)


# ──────────────────────────────────────────────────────────
# Probe training helpers
# ──────────────────────────────────────────────────────────

def train_probe_epoch(model, loader, criterion, optimizer, device):
    """One epoch of supervised training for the probe head."""
    model.train()
    correct, total, total_loss = 0, 0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_probe(model, loader, criterion, device):
    """Evaluate the probe on a loader; returns (avg_loss, accuracy)."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────

def visualize_reconstructions(
    model, loader, device, n: int = 8,
    title: str = "Reconstructions",
    save_path: str = "outputs/reconstructions.png",
):
    """
    Side-by-side grid: top row = originals, bottom row = reconstructions.

    Handles both AE (model returns a tensor) and VAE (model returns a tuple
    where the first element is the reconstruction tensor).
    """
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs[:n].to(device)
    with torch.no_grad():
        out = model(imgs)
    recons = out[0] if isinstance(out, tuple) else out

    imgs_np   = imgs.cpu().permute(0, 2, 3, 1).numpy()
    recons_np = recons.cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(np.clip(imgs_np[i], 0, 1))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=9)
        axes[1, i].imshow(np.clip(recons_np[i], 0, 1))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Recon", fontsize=9)

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved reconstruction plot → {save_path}")


def visualize_probe_predictions(
    probe, loader, device, n_rows: int = 4, n_cols: int = 8,
    title: str = "Linear Probe – Test Predictions",
    save_path: str = "outputs/probe_predictions.png",
    denorm_fn=None,
):
    """
    Grid of test images annotated with true (T) and predicted (P) class names.
    Green = correct, red = wrong.

    denorm_fn : optional callable (tensor → tensor) applied before display to
                undo channel normalisation (e.g. for SimCLR / ResNet inputs).
    """
    probe.eval()
    all_imgs, all_true, all_pred = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = probe(imgs.to(device)).argmax(1).cpu()
            all_imgs.append(imgs.cpu())
            all_true.extend(labels.tolist())
            all_pred.extend(preds.tolist())
            if len(all_true) >= n_rows * n_cols:
                break

    imgs_t = torch.cat(all_imgs, 0)
    if denorm_fn is not None:
        imgs_t = denorm_fn(imgs_t)
    imgs_np = imgs_t.permute(0, 2, 3, 1).numpy()

    n = n_rows * n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.5 * n_rows))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(np.clip(imgs_np[i], 0, 1))
        ax.axis("off")
        color = "green" if all_true[i] == all_pred[i] else "red"
        ax.set_title(
            f"T:{CIFAR10_CLASSES[all_true[i]]}\nP:{CIFAR10_CLASSES[all_pred[i]]}",
            fontsize=7, color=color,
        )

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved prediction grid → {save_path}")
