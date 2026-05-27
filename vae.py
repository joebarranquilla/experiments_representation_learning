"""
Convolutional Variational Autoencoder (VAE) trained on CIFAR-10.

Differences from autoencoder.py
--------------------------------
- Encoder outputs (mu, log_var) instead of a deterministic z.
- Training loss = MSE reconstruction + beta * KL divergence.
- beta-VAE annealing: beta starts at 0 and linearly ramps to beta_max over
  the first `beta_warmup` epochs so the model learns to reconstruct first.
- Encoder saves mu (deterministic) for linear probing — avoids stochasticity
  at inference time, giving stable features.

Usage
-----
    python vae.py                 # train VAE, save, then run linear probing
    python vae.py --probe-only    # skip VAE training, load saved encoder
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
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

class VAEEncoder(nn.Module):
    """
    Conv encoder: 3×32×32 → (mu, log_var) each of shape (B, latent_dim).

    Wider channels (64/128/256 vs old 32/64/128) give the encoder more
    capacity.  BatchNorm + LeakyReLU after each conv for stable training.
    log_var is clamped to [-4, 4] to prevent exp(log_var) blowing up KL.
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,   64,  kernel_size=4, stride=2, padding=1),  # 16×16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1),  # 8×8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 4×4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        flat_dim = 256 * 4 * 4
        self.fc_mu      = nn.Linear(flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h).clamp(-4.0, 4.0)  # numerical stability
        return mu, log_var


class VAEDecoder(nn.Module):
    """
    Conv decoder: latent_dim-d vector → 3×32×32 image.
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
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    """
    Full VAE = VAEEncoder + reparameterisation + VAEDecoder.

    forward() returns (reconstruction, mu, log_var).
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    def reparameterise(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at eval time

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterise(mu, log_var)
        return self.decoder(z), mu, log_var

    def encode_mu(self, x):
        """Return only mu — used by LinearProbe at inference."""
        mu, _ = self.encoder(x)
        return mu


# ──────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────

def vae_loss(recon, target, mu, log_var, beta: float = 1.0, free_bits: float = 0.5):
    """
    ELBO loss: MSE reconstruction + beta * KL divergence.

    Free bits: each latent dimension is given a minimum KL budget of
    `free_bits` nats.  Dimensions already above the budget contribute
    normally; dimensions below it are clamped, which prevents the encoder
    from collapsing those dimensions to the prior (posterior collapse).
    """
    recon_loss = F.mse_loss(recon, target, reduction="mean")

    # Per-dimension KL: shape (B, latent_dim)
    kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
    # Average over batch first → (latent_dim,), then apply free-bits floor
    kl = torch.clamp(kl_per_dim.mean(0), min=free_bits).mean()

    return recon_loss + beta * kl, recon_loss.item(), kl.item()


# ──────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────

def train_vae_epoch(model, loader, optimizer, device, beta: float):
    model.train()
    total, total_recon, total_kl = 0.0, 0.0, 0.0
    n = len(loader.dataset)
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon, mu, log_var = model(imgs)
        loss, r, k = vae_loss(recon, imgs, mu, log_var, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = imgs.size(0)
        total       += loss.item() * bs
        total_recon += r * bs
        total_kl    += k * bs
    return total / n, total_recon / n, total_kl / n


@torch.no_grad()
def eval_vae(model, loader, device, beta: float):
    model.eval()
    total, total_recon, total_kl = 0.0, 0.0, 0.0
    n = len(loader.dataset)
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon, mu, log_var = model(imgs)
        loss, r, k = vae_loss(recon, imgs, mu, log_var, beta)
        bs = imgs.size(0)
        total       += loss.item() * bs
        total_recon += r * bs
        total_kl    += k * bs
    return total / n, total_recon / n, total_kl / n


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main(probe_only: bool = False):
    # ── Hyperparameters ──────────────────────────────────
    latent_dim   = 256
    vae_epochs   = 60
    probe_epochs = 100
    batch_size   = 128
    vae_lr       = 1e-3
    probe_lr     = 1e-3
    probe_hidden = [512]        # [] for pure linear probe
    weight_decay = 1e-4
    patience     = 15
    beta_max     = 0.5          # final KL weight (lower = less regularisation pressure)
    beta_warmup  = 20           # epochs to linearly ramp beta from 0 → beta_max
    save_dir     = "checkpoints"
    vae_ckpt     = os.path.join(save_dir, "vae_full.pt")
    encoder_ckpt = os.path.join(save_dir, "vae_encoder.pt")
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir,  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    print(f"Device: {device}")

    # Augmented loader for VAE training, clean loaders for probe eval
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, augment_train=True
    )

    # ── 1. Train (or load) VAE ───────────────────────────
    vae = VAE(latent_dim).to(device)

    if probe_only:
        if not os.path.exists(vae_ckpt):
            raise FileNotFoundError(f"No checkpoint found at {vae_ckpt}. Run without --probe-only first.")
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        print(f"Loaded VAE from {vae_ckpt}")
    else:
        optimizer = optim.Adam(vae.parameters(), lr=vae_lr, weight_decay=weight_decay)

        print(f"\n── Training VAE for up to {vae_epochs} epochs ──")
        best_val_loss = float("inf")
        best_state    = None
        patience_ctr  = 0
        t0 = time.time()

        for epoch in range(vae_epochs):
            epoch_t0 = time.time()
            # Beta annealing: 0 → beta_max over beta_warmup epochs
            beta = min(beta_max, beta_max * (epoch + 1) / beta_warmup)

            train_loss, train_r, train_kl = train_vae_epoch(vae, train_loader, optimizer, device, beta)
            val_loss,   val_r,   val_kl   = eval_vae(vae, val_loader, device, beta)
            epoch_elapsed = time.time() - epoch_t0

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{vae_epochs} | beta={beta:.2f} | "
                      f"Train Loss: {train_loss:.4f} (R:{train_r:.4f} KL:{train_kl:.4f}) | "
                      f"Val Loss: {val_loss:.4f} (R:{val_r:.4f} KL:{val_kl:.4f})")
            # Always print per-epoch duration for debugging/perf monitoring
            print(f"    Epoch time: {epoch_elapsed:.2f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in vae.state_dict().items()}
                patience_ctr  = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                    break

        vae.load_state_dict(best_state)
        elapsed = time.time() - t0
        print(f"VAE training done in {elapsed:.1f}s | Best val Loss: {best_val_loss:.4f}")

        torch.save(vae.state_dict(),          vae_ckpt)
        torch.save(vae.encoder.state_dict(),  encoder_ckpt)
        print(f"Saved full VAE  → {vae_ckpt}")
        print(f"Saved encoder   → {encoder_ckpt}")

        visualize_reconstructions(vae, val_loader, device,
                                  title="VAE Reconstructions",
                                  save_path="outputs/vae_reconstructions.png")

    # ── 2. Linear probing ────────────────────────────────
    print(f"\n── Linear Probing (frozen VAE encoder, uses mu) ──")
    probe = LinearProbe(vae, latent_dim,
                        encode_fn=vae.encode_mu,
                        hidden_dims=probe_hidden).to(device)
    probe_criterion = nn.CrossEntropyLoss()
    probe_optimizer = optim.Adam(probe.head.parameters(), lr=probe_lr, weight_decay=weight_decay)
    probe_scheduler = optim.lr_scheduler.CosineAnnealingLR(probe_optimizer, T_max=probe_epochs)

    best_val_loss    = float("inf")
    best_probe_state = None
    patience_ctr     = 0

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

    _, test_acc = eval_probe(probe, test_loader, probe_criterion, device)
    print(f"\n── Results ──────────────────────────────────────")
    print(f"  Best Val Loss  : {best_val_loss:.4f}")
    print(f"  Test Accuracy  : {test_acc:.4f} ({test_acc*100:.1f}%)")

    visualize_probe_predictions(probe, test_loader, device,
                                title="VAE Linear Probe – Test Predictions",
                                save_path="outputs/vae_probe_predictions.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-only", action="store_true",
                        help="Skip VAE training; load saved checkpoint and run linear probing only.")
    args = parser.parse_args()
    main(probe_only=args.probe_only)
