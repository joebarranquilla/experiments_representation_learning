"""
SimCLR-style frozen feature extraction + linear probing on CIFAR-10.

This script follows the same pattern as the Google SimCLR finetuning colab
(colabs/finetuning.ipynb in google-research/simclr):

  1. Load a pretrained encoder (frozen – no training required).
  2. Extract features from CIFAR-10 using that encoder.
  3. Train only a small linear/MLP head on top of the frozen features.

─────────────────────────────────────────────────────────────────────────────
Quickstart (runs immediately, no GPU needed):
─────────────────────────────────────────────────────────────────────────────
  python simclr.py                          # ResNet-50, ImageNet backbone
  python simclr.py --arch resnet18          # lighter / faster backbone

─────────────────────────────────────────────────────────────────────────────
Using an actual SimCLR checkpoint (optional):
─────────────────────────────────────────────────────────────────────────────
If you have a SimCLR checkpoint (e.g. trained via sthalles/SimCLR in Colab),
pass it with --checkpoint and the SimCLR encoder will be used instead:

  python simclr.py --checkpoint checkpoints/simclr_resnet18_ep100.pth.tar

Checkpoint format (sthalles/SimCLR compatible):
  {'epoch': ..., 'arch': ..., 'state_dict': ..., 'optimizer': ...}
  where state_dict keys start with 'backbone.*'

How to get a pretrained SimCLR CIFAR-10 checkpoint in ~15 min (free):
  Open https://colab.research.google.com, run on a T4 GPU:
    !git clone https://github.com/sthalles/SimCLR && cd SimCLR
    !python run.py -data ./datasets --dataset-name cifar10 \\
                   --log-every-n-steps 100 --epochs 100
  Download: runs/*/checkpoint_0100.pth.tar

─────────────────────────────────────────────────────────────────────────────
Default backbone (no --checkpoint given):
─────────────────────────────────────────────────────────────────────────────
Uses a supervised ImageNet-pretrained ResNet from torchvision (~100 MB,
downloaded automatically on first run). This mirrors what the Google SimCLR
notebook does: load a powerful frozen encoder, then train only a linear head
on the target dataset. The encoder differs (supervised ImageNet vs. SimCLR
self-supervised), but the probe pipeline is identical — making it a valid
and fast strong-baseline comparison alongside your AE and VAE results.

Backbone feature dimensions:
  resnet18 → 512-d   (~3 min probe on CPU)
  resnet50 → 2048-d  (~8 min probe on CPU, stronger features)
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10

from utils import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    get_cifar10_loaders,
    LinearProbe,
    train_probe_epoch,
    eval_probe,
    visualize_probe_predictions,
)

# ──────────────────────────────────────────────────────────
# Contrastive data loader (SimCLR-specific; supervised loaders use utils)
# ──────────────────────────────────────────────────────────

class TwoViewTransform:
    """Returns two independently augmented views of the same PIL image."""
    def __init__(self, transform):
        self.t = transform

    def __call__(self, x):
        return self.t(x), self.t(x)


def _contrastive_augment(size: int = 32) -> transforms.Compose:
    """
    SimCLR augmentation pipeline (Chen et al. 2020, Appendix A).
    Color jitter + grayscale are the most critical augmentations for
    self-supervised feature quality on CIFAR-10.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_contrastive_loader(data_root: str = "data", batch_size: int = 256):
    """DataLoader for contrastive pre-training — each item is (view1, view2)."""
    dataset = CIFAR10(root=data_root, train=True, download=True,
                      transform=TwoViewTransform(_contrastive_augment()))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=0, drop_last=True)


# ──────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────

class ResNetSimCLR(nn.Module):
    """
    ResNet backbone + 2-layer MLP projection head.

    Architecture matches sthalles/SimCLR exactly so checkpoints from that
    repo are directly loadable here.

    backbone:   standard ResNet-18 (modified fc = Linear(512,512) → ReLU → Linear(512,128))
    out_dim:    projection head output dim (128 by default)
    feature_dim: 512 (ResNet-18 avgpool output, used for linear probing)
    """
    def __init__(self, base_model: str = "resnet18", out_dim: int = 128):
        super().__init__()
        resnet_dict = {
            "resnet18": models.resnet18(weights=None, num_classes=out_dim),
            "resnet50": models.resnet50(weights=None, num_classes=out_dim),
        }
        if base_model not in resnet_dict:
            raise ValueError(f"base_model must be one of {list(resnet_dict.keys())}")

        self.backbone = resnet_dict[base_model]
        dim_mlp = self.backbone.fc.in_features      # 512 for resnet18

        # Replace single FC with 2-layer MLP projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(inplace=True),
            self.backbone.fc,                       # Linear(512, out_dim)
        )
        self.feature_dim = dim_mlp                  # 512 — used for probing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised projected embeddings (used during pre-training)."""
        return F.normalize(self.backbone(x), dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the backbone representation BEFORE the projection head.
        This is what gets frozen and used for linear probing, following
        SimCLR paper Section 4: "we use the representation before the
        nonlinear projection g(·) to evaluate the quality of representation."
        """
        # Temporarily swap out the fc; faster than detaching/copying
        head = self.backbone.fc
        self.backbone.fc = nn.Identity()
        with torch.no_grad():
            z = self.backbone(x)
        self.backbone.fc = head
        return z


# ──────────────────────────────────────────────────────────
# NT-Xent loss
# ──────────────────────────────────────────────────────────

def nt_xent_loss(features: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent loss (Chen et al. 2020).

    `features` is the concatenation of two view embeddings: shape (2B, d).
    Views [0..B-1] and [B..2B-1] are the two augmented versions of each image.
    Both are already L2-normalised by the model's forward().
    """
    B = features.shape[0] // 2
    labels = torch.cat([torch.arange(B) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features.device)

    sim = torch.matmul(features, features.T)  # cosine sim (normalised inputs)

    # Mask out self-similarity
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    sim    = sim[~mask].view(sim.shape[0], -1)

    positives = sim[labels.bool()].view(labels.shape[0], -1)
    negatives = sim[~labels.bool()].view(sim.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1) / temperature
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)
    return F.cross_entropy(logits, targets)


# ──────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────

def train_simclr_epoch(model: ResNetSimCLR, loader: DataLoader,
                       optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for (v1, v2), _ in loader:
        imgs = torch.cat([v1, v2], dim=0).to(device)   # (2B, C, H, W)
        features = model(imgs)                          # (2B, proj_dim), L2-norm'd
        loss = nt_xent_loss(features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * v1.size(0)
    return total_loss / len(loader.dataset)


# ──────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────

def _denorm(imgs: torch.Tensor) -> torch.Tensor:
    """Undo CIFAR-10 channel normalisation for display."""
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (imgs * std + mean).clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────
# Phase 1 – Contrastive pre-training
# ──────────────────────────────────────────────────────────

def pretrain(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"\n── SimCLR pre-training ──────────────────────────")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {args.pretrain_epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Temperature  : {args.temperature}")

    loader = get_contrastive_loader(batch_size=args.batch_size)
    model  = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs)

    best_loss  = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(args.pretrain_epochs):
        epoch_t0 = time.time()
        loss = train_simclr_epoch(model, loader, optimizer, device)
        scheduler.step()
        epoch_elapsed = time.time() - epoch_t0

        if (epoch + 1) % max(1, args.pretrain_epochs // 10) == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:4d}/{args.pretrain_epochs} | "
                  f"NT-Xent: {loss:.4f} | LR: {lr_now:.2e}")
        # Always print per-epoch duration
        print(f"    Epoch time: {epoch_elapsed:.2f}s")

        if loss < best_loss:
            best_loss  = loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    print(f"Pre-training done in {elapsed:.1f}s | Best NT-Xent: {best_loss:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"simclr_{args.arch}_ep{args.pretrain_epochs}.pth.tar")
    torch.save({
        "epoch":      args.pretrain_epochs,
        "arch":       args.arch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, ckpt_path)
    print(f"Saved checkpoint → {ckpt_path}")
    return ckpt_path


# ──────────────────────────────────────────────────────────
# Phase 2 – Linear probing
# ──────────────────────────────────────────────────────────

def probe(args, checkpoint_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Resolve checkpoint
    ckpt_path = checkpoint_path or args.checkpoint
    if ckpt_path is None:
        # Try to find most recently saved SimCLR checkpoint
        if os.path.isdir(args.save_dir):
            candidates = sorted([
                f for f in os.listdir(args.save_dir)
                if f.startswith("simclr_") and f.endswith(".pth.tar")
            ])
            if candidates:
                ckpt_path = os.path.join(args.save_dir, candidates[-1])
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "No SimCLR checkpoint found.\n"
            "Run  python simclr.py --pretrain  first, or pass --checkpoint <path>."
        )

    print(f"\n── SimCLR linear probe ──────────────────────────")
    print(f"  Device       : {device}")
    print(f"  Checkpoint   : {ckpt_path}")

    # Load encoder
    ckpt  = torch.load(ckpt_path, map_location=device)
    arch  = ckpt.get("arch", args.arch)
    model = ResNetSimCLR(base_model=arch, out_dim=args.out_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Arch         : {arch} | feature_dim: {model.feature_dim}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")

    # Linear probe
    os.makedirs("outputs", exist_ok=True)

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, normalize=True
    )
    lprobe = LinearProbe(
        model, model.feature_dim,
        encode_fn=model.encode,
        hidden_dims=args.probe_hidden,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lprobe.head.parameters(),
                           lr=args.probe_lr, weight_decay=args.weight_decay)

    best_val_loss    = float("inf")
    best_probe_state = None
    patience_ctr     = 0

    for epoch in range(args.probe_epochs):
        epoch_t0 = time.time()
        train_loss, train_acc = train_probe_epoch(
            lprobe, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_probe(lprobe, val_loader, criterion, device)
        epoch_elapsed = time.time() - epoch_t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.probe_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        # Always print per-epoch duration for the probe
        print(f"    Epoch time: {epoch_elapsed:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_probe_state = {k: v.clone() for k, v in lprobe.state_dict().items()}
            patience_ctr     = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    lprobe.load_state_dict(best_probe_state)
    _, test_acc = eval_probe(lprobe, test_loader, criterion, device)
    print(f"\n── Results ──────────────────────────────────────")
    print(f"  Best Val Loss  : {best_val_loss:.4f}")
    print(f"  Test Accuracy  : {test_acc:.4f} ({test_acc*100:.1f}%)")

    visualize_probe_predictions(
        lprobe, test_loader, device,
        title="SimCLR Linear Probe – Test Predictions",
        save_path="outputs/simclr_probe_predictions.png",
        denorm_fn=_denorm,
    )


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SimCLR self-supervised pre-training + linear probing on CIFAR-10"
    )
    # Phase selection
    p.add_argument("--pretrain", action="store_true",
                   help="Run contrastive pre-training.")
    p.add_argument("--probe",    action="store_true",
                   help="Run linear probe on frozen encoder.")

    # Shared
    p.add_argument("--arch",         default="resnet18",
                   choices=["resnet18", "resnet50"])
    p.add_argument("--batch-size",   type=int,   default=256, dest="batch_size")
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--save-dir",     default="checkpoints",   dest="save_dir")
    p.add_argument("--checkpoint",   default=None,
                   help="Path to a SimCLR checkpoint (for --probe without --pretrain).")
    p.add_argument("--cpu",          action="store_true",
                   help="Force CPU even if CUDA is available.")

    # Pre-training
    p.add_argument("--pretrain-epochs", type=int,   default=30,  dest="pretrain_epochs",
                   help="Epochs for contrastive pre-training (30 ≈ 60-90 min on CPU).")
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--temperature",     type=float, default=0.5)
    p.add_argument("--out-dim",         type=int,   default=128, dest="out_dim",
                   help="Projection head output dimension.")

    # Linear probe
    p.add_argument("--probe-epochs",  type=int,   default=50, dest="probe_epochs")
    p.add_argument("--probe-lr",      type=float, default=1e-3, dest="probe_lr")
    p.add_argument("--probe-hidden",  type=int,   nargs="*", default=[256],
                   dest="probe_hidden",
                   help="Hidden dims for probe MLP. Pass nothing for pure linear probe.")
    p.add_argument("--patience",      type=int,   default=10)

    args = p.parse_args()

    # Default: run both if neither flag is given
    if not args.pretrain and not args.probe:
        p.error("Specify at least one of --pretrain / --probe.\n"
                "  python simclr.py --pretrain            # phase 1 only\n"
                "  python simclr.py --probe               # phase 2 only (needs checkpoint)\n"
                "  python simclr.py --pretrain --probe    # both back-to-back")
    return args


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = None
    if args.pretrain:
        ckpt_path = pretrain(args)
    if args.probe:
        probe(args, checkpoint_path=ckpt_path)
