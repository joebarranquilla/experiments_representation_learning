"""
Convolutional Autoencoder trained on CIFAR-10.

Workflow
--------
1. Train the AE on reconstruction (MSE loss).
2. Save the full model and the encoder weights separately.
3. Linear probing: freeze the encoder, attach a small MLP head, and train on
   class labels — mirrors the PCA-MLP pipeline in pca_mlp_classifier.py.

Usage
-----
    python autoencoder.py            # train AE, save, then run linear probing
    python autoencoder.py --probe-only  # skip AE training, load saved encoder
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils import (
    get_cifar10_loaders,
    LinearProbe,
    train_probe_epoch,
    eval_probe,
    visualize_reconstructions,
    visualize_probe_predictions,
)

# ──────────────────────────────────────────────────────────
# Data  (see utils.get_cifar10_loaders)
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
# Model components
# ──────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Conv encoder: 3×32×32  →  latent_dim-d vector.

    Architecture (stride-2 convs + BatchNorm for stable, rich features):
      3×32×32 → 64×16×16 → 128×8×8 → 256×4×4 → flatten → FC(latent_dim)

    Wider channels (64/128/256 vs old 32/64/128) give the encoder more
    capacity to capture class-discriminative structure rather than just
    low-level texture.  BatchNorm stabilises training and acts as implicit
    regularisation, producing smoother latent spaces that are easier for
    a linear classifier to separate.
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,   64,  kernel_size=4, stride=2, padding=1),  # 16×16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1),  # 8×8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 4×4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class Decoder(nn.Module):
    """
    Conv decoder: latent_dim-d vector  →  3×32×32 image.

    Mirrors the wider encoder with transposed convolutions.
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8×8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),  # 16×16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  3,   kernel_size=4, stride=2, padding=1),  # 32×32
            nn.Sigmoid(),  # map to [0, 1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.deconv(h)


class Autoencoder(nn.Module):
    """Full AE = Encoder + Decoder."""
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ──────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────

def train_ae_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon = model(imgs)
        loss = criterion(recon, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_ae(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon = model(imgs)
        total_loss += criterion(recon, imgs).item() * imgs.size(0)
    return total_loss / len(loader.dataset)


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main(probe_only: bool = False):
    # ── Hyperparameters ──────────────────────────────────
    latent_dim      = 256
    ae_epochs       = 60
    probe_epochs    = 100
    batch_size      = 128
    ae_lr           = 5e-4
    probe_lr        = 1e-3
    probe_hidden    = [512]          # [] for pure linear probe
    weight_decay    = 1e-4
    patience        = 15             # early stopping patience (val loss)
    save_dir        = "checkpoints"
    ae_ckpt         = os.path.join(save_dir, "ae_full.pt")
    encoder_ckpt    = os.path.join(save_dir, "ae_encoder.pt")
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir,   exist_ok=True)
    os.makedirs("outputs",  exist_ok=True)
    print(f"Device: {device}")

    # Augmented loader for AE training, clean loaders for probe eval
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, augment_train=True
    )

    # ── 1. Train (or load) Autoencoder ──────────────────
    ae = Autoencoder(latent_dim).to(device)

    if probe_only:
        if not os.path.exists(ae_ckpt):
            raise FileNotFoundError(f"No checkpoint found at {ae_ckpt}. Run without --probe-only first.")
        ae.load_state_dict(torch.load(ae_ckpt, map_location=device))
        print(f"Loaded AE from {ae_ckpt}")
    else:
        ae_criterion = nn.MSELoss()
        ae_optimizer = optim.Adam(ae.parameters(), lr=ae_lr, weight_decay=weight_decay)

        print(f"\n── Training Autoencoder for up to {ae_epochs} epochs ──")
        best_val_loss = float("inf")
        best_state    = None
        patience_ctr  = 0
        t0 = time.time()

        for epoch in range(ae_epochs):
            epoch_t0 = time.time()
            train_loss = train_ae_epoch(ae, train_loader, ae_criterion, ae_optimizer, device)
            val_loss   = eval_ae(ae, val_loader, ae_criterion, device)
            epoch_elapsed = time.time() - epoch_t0

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{ae_epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
            # Always print per-epoch duration for debugging/perf monitoring
            print(f"    Epoch time: {epoch_elapsed:.2f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in ae.state_dict().items()}
                patience_ctr  = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                    break

        ae.load_state_dict(best_state)
        elapsed = time.time() - t0
        print(f"AE training done in {elapsed:.1f}s | Best val MSE: {best_val_loss:.4f}")

        # Save
        torch.save(ae.state_dict(),         ae_ckpt)
        torch.save(ae.encoder.state_dict(), encoder_ckpt)
        print(f"Saved full AE   → {ae_ckpt}")
        print(f"Saved encoder   → {encoder_ckpt}")

        # Visualise reconstructions
        visualize_reconstructions(ae, val_loader, device,
                                  title="AE Reconstructions",
                                  save_path="outputs/ae_reconstructions.png")

    # ── 2. Linear probing ────────────────────────────────
    print(f"\n── Linear Probing (frozen encoder) ──")
    probe = LinearProbe(ae.encoder, latent_dim, hidden_dims=probe_hidden).to(device)
    probe_criterion = nn.CrossEntropyLoss()
    probe_optimizer = optim.Adam(probe.head.parameters(), lr=probe_lr, weight_decay=weight_decay)
    probe_scheduler = optim.lr_scheduler.CosineAnnealingLR(probe_optimizer, T_max=probe_epochs)

    best_val_loss = float("inf")
    best_probe_state = None
    patience_ctr = 0

    for epoch in range(probe_epochs):
        epoch_t0 = time.time()
        train_loss, train_acc = train_probe_epoch(probe, train_loader, probe_criterion, probe_optimizer, device)
        val_loss,   val_acc   = eval_probe(probe, val_loader, probe_criterion, device)
        epoch_elapsed = time.time() - epoch_t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{probe_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        # Always print per-epoch duration for the probe
        print(f"    Epoch time: {epoch_elapsed:.2f}s")

        probe_scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_probe_state = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_ctr     = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    probe.load_state_dict(best_probe_state)

    # Final evaluation
    _, test_acc = eval_probe(probe, test_loader, probe_criterion, device)
    print(f"\n── Results ──────────────────────────────────────")
    print(f"  Best Val Loss  : {best_val_loss:.4f}")
    print(f"  Test Accuracy  : {test_acc:.4f} ({test_acc*100:.1f}%)")

    # Prediction grid
    visualize_probe_predictions(probe, test_loader, device,
                                title="AE Linear Probe – Test Predictions",
                                save_path="outputs/ae_probe_predictions.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-only", action="store_true",
                        help="Skip AE training; load saved checkpoint and run linear probing only.")
    args = parser.parse_args()
    main(probe_only=args.probe_only)
