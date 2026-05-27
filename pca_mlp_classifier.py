import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
from utils import visualize_probe_predictions


def load_and_preprocess_cifar10(data_root="data", val_fraction=0.1):
    """
    Load CIFAR-10, flatten images, and split into train/val/test.

    Mirrors utils.get_cifar10_loaders: the 50K training set is split with a
    fixed seed (42) via torch.randperm, and the official 10K test set is kept
    separate.  No sklearn train_test_split is used.

    Parameters
    ----------
    data_root : str
        Directory where CIFAR-10 is downloaded / cached.
    val_fraction : float
        Fraction of the 50K training set to hold out for validation.

    Returns
    -------
    X_train, X_val, X_test : float32 ndarrays, shape (N, 3072)
    y_train, y_val, y_test : int64 ndarrays, shape (N,)
    """
    to_tensor = transforms.ToTensor()

    train_ds = CIFAR10(root=data_root, train=True,  download=True, transform=to_tensor)
    test_ds  = CIFAR10(root=data_root, train=False, download=True, transform=to_tensor)

    def _dataset_to_arrays(ds):
        X, y = [], []
        for img, label in ds:
            X.append(img.numpy().transpose(1, 2, 0).reshape(-1))
            y.append(label)
        return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64)

    print("Loading CIFAR-10...")
    X_all, y_all = _dataset_to_arrays(train_ds)
    X_test, y_test = _dataset_to_arrays(test_ds)

    # Reproducible train/val split — identical seed to utils.py
    n_val   = int(len(X_all) * val_fraction)
    n_train = len(X_all) - n_val
    indices = torch.randperm(
        len(X_all), generator=torch.Generator().manual_seed(42)
    ).tolist()
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val,   y_val   = X_all[val_idx],   y_all[val_idx]

    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test



class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron with ReLU activations.
    """
    def __init__(self, input_dim, hidden_dims=[256, 128], num_classes=10):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PCAClassifier(nn.Module):
    """
    Wraps a fitted sklearn PCA + a trained MLP so the combined model accepts
    raw image tensors of shape (B, 3, 32, 32) in [0, 1].

    This makes it a drop-in for visualize_probe_predictions from utils.py,
    which calls probe(imgs) where imgs come from a standard CIFAR-10 loader.
    """

    def __init__(self, pca, mlp: nn.Module):
        super().__init__()
        self.pca = pca          # sklearn PCA (not a nn.Module)
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 32, 32)  →  flatten to (B, 3072)  →  PCA  →  MLP logits
        flat  = x.cpu().numpy().transpose(0, 2, 3, 1).reshape(x.shape[0], -1)
        feats = torch.from_numpy(self.pca.transform(flat).astype(np.float32))
        feats = feats.to(next(self.mlp.parameters()).device)
        return self.mlp(feats)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(train_loader.dataset)


def evaluate(model, data_loader, device, return_loss=False, criterion=None):
    """Evaluate model accuracy and optionally loss."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            if return_loss and criterion is not None:
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * X_batch.size(0)

    acc = correct / total
    if return_loss and criterion is not None:
        loss = total_loss / total
        return acc, loss
    return acc


# visualize_predictions is provided by utils.visualize_probe_predictions via
# the PCAClassifier wrapper — no local duplicate needed.


def main():
    # Hyperparameters
    n_components = 80  # Increased PCA components to capture more variance
    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-3
    hidden_dims = [64]  # Simplified: single small hidden layer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"PCA n_components: {n_components}")

    # Load and preprocess data (train/val/test)
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_cifar10(
        data_root="data", val_fraction=0.1
    )

    # Apply PCA
    print(f"\nFitting PCA with {n_components} components...")
    # whiten=True divides each PC by its std (eigenvalue sqrt), normalising the
    # feature scale and making full use of PCA's variance decomposition.
    pca = PCA(n_components=n_components, whiten=True)
    X_train_pca = pca.fit_transform(X_train).astype(np.float32)
    X_val_pca = pca.transform(X_val).astype(np.float32) if X_val is not None else None
    X_test_pca = pca.transform(X_test).astype(np.float32)

    print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  Train shape after PCA: {X_train_pca.shape}")
    if X_val_pca is not None:
        print(f"  Val shape after PCA: {X_val_pca.shape}")
    print(f"  Test shape after PCA: {X_test_pca.shape}")
    # Optional augmentation in PCA space: add Gaussian noise to PCA features
    do_augment = True
    noise_sigma = 0.05  # Reduced noise (was 0.1)
    n_copies = 1       # how many augmented copies to create per sample
    if do_augment and n_copies > 0:
        print(f"Applying Gaussian noise augmentation in PCA space: sigma={noise_sigma}, copies={n_copies}")
        n_train = X_train_pca.shape[0]
        # generate noise of shape (n_copies * n_train, n_components)
        noise = np.random.normal(loc=0.0, scale=noise_sigma, size=(n_copies * n_train, X_train_pca.shape[1])).astype(np.float32)
        X_aug_list = []
        y_aug_list = []
        for i in range(n_copies):
            start = i * n_train
            X_aug = X_train_pca + noise[start:start + n_train]
            X_aug_list.append(X_aug)
            y_aug_list.append(y_train.copy())

        X_aug_all = np.vstack(X_aug_list)
        y_aug_all = np.concatenate(y_aug_list)

        # Append augmented data
        X_train_pca = np.vstack([X_train_pca, X_aug_all])
        y_train = np.concatenate([y_train, y_aug_all])
        print(f"  Augmented train shape after PCA: {X_train_pca.shape}")

    # Create data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train_pca), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val_pca), torch.from_numpy(y_val).long()) if X_val_pca is not None else None
    test_dataset = TensorDataset(torch.from_numpy(X_test_pca), torch.from_numpy(y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = MLP(input_dim=n_components, hidden_dims=hidden_dims, num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # multi-class cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print(f"\nModel architecture:")
    print(f"  Input: {n_components} (PCA features)")
    print(f"  Hidden: {hidden_dims}")
    print(f"  Output: 10 (classes)")
    print(f"\nTraining for {num_epochs} epochs...")

    # Training loop with early stopping on validation loss
    early_stop_patience = 15
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_loss = evaluate(model, val_loader, device, return_loss=True, criterion=criterion) if val_loader is not None else (None, None)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            msg = f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f}"
            if val_acc is not None:
                msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            print(msg)

        # Early stopping: track best val_loss and save model state
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if val_loss is not None and (epoch + 1 - best_epoch) >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f} at epoch {best_epoch})")
            break

    elapsed = time.time() - start_time

    # Restore best model before final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation (test acc printed only here)
    train_acc = evaluate(model, train_loader, device)
    val_acc = evaluate(model, val_loader, device) if val_loader is not None else None
    test_acc = evaluate(model, test_loader, device)

    print(f"\nTraining completed in {elapsed:.2f}s (best epoch: {best_epoch})")
    print(f"Final Train Accuracy: {train_acc:.4f}")
    if val_acc is not None:
        print(f"Final Val Accuracy:   {val_acc:.4f}")
    print(f"Final Test Accuracy:  {test_acc:.4f}")

    # ── Visualise predictions using the shared utils helper ──────────────────
    # Reconstruct image tensors (N, 3, 32, 32) from the flat arrays so the
    # standard image DataLoader that visualize_probe_predictions expects works.
    import os
    os.makedirs("outputs", exist_ok=True)

    X_test_imgs = torch.from_numpy(
        X_test.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)  # (N, C, H, W)
    )
    viz_loader = DataLoader(
        TensorDataset(X_test_imgs, torch.from_numpy(y_test).long()),
        batch_size=64, shuffle=False,
    )
    pca_wrapper = PCAClassifier(pca, model)
    visualize_probe_predictions(
        pca_wrapper, viz_loader, device,
        title='PCA + MLP – Test Predictions',
        save_path='outputs/pca_mlp_predictions.png',
    )


if __name__ == "__main__":
    main()
