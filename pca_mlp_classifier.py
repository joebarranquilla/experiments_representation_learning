import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt


def load_and_preprocess_cifar10(n_samples=None, test_size=0.2, val_size=0.2):
    """
    Load CIFAR-10, flatten images, and split into train/val/test.

    Parameters
    ----------
    n_samples : int or None
        Number of samples to use (subsample for speed). None uses all.
    test_size : float
        Fraction of data to use for testing.
    val_size : float
        Fraction of the full dataset to use for validation.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : ndarrays
        Flattened image data and labels.
    """
    print("Loading CIFAR-10...")
    dataset = CIFAR10(root="data", train=True, download=True, transform=ToTensor())

    # Convert to numpy and flatten images
    X_list = []
    y_list = []
    for img, label in dataset:
        arr = img.numpy()
        arr = np.transpose(arr, (1, 2, 0)).reshape(-1)
        X_list.append(arr)
        y_list.append(label)

    X = np.vstack(X_list)
    y = np.array(y_list)

    # Subsample if requested (use global RNG)
    if n_samples is not None and n_samples < X.shape[0]:
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y
    )

    # Now split the remaining data into train and validation
    if val_size is not None and val_size > 0:
        val_rel = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_rel, stratify=y_temp
        )
    else:
        X_train, y_train = X_temp, y_temp
        X_val, y_val = None, None

    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    if X_val is not None:
        print(f"  Val: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

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


def visualize_predictions(model, X_test_flat, X_test_pca, y_test, device, classes, num_images=16):
    """Plot a grid of test images with true and predicted labels.

    Parameters
    - model: torch model that expects PCA features input (MLP)
    - X_test_flat: numpy array of flattened images (H,W,C flattened)
    - X_test_pca: numpy array of PCA-transformed features for test set
    - y_test: numpy array of integer labels
    - device: torch device
    - classes: list of class names
    - num_images: how many images to display (will form roughly square grid)
    """
    model.eval()
    n = X_test_flat.shape[0]
    num_images = min(num_images, n)
    idxs = np.random.choice(n, num_images, replace=False)

    # Prepare inputs for model (PCA features)
    X_sel_pca = X_test_pca[idxs]
    X_tensor = torch.from_numpy(X_sel_pca).float().to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Plot grid
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= num_images:
            ax.axis('off')
            continue

        idx = idxs[i]
        img_flat = X_test_flat[idx]
        # reshape back to H,W,C where H=W=32
        img = img_flat.reshape(32, 32, 3)

        true_lbl = int(y_test[idx])
        pred_lbl = int(preds[i])

        ax.imshow(img)
        title = f"T: {classes[true_lbl]}\nP: {classes[pred_lbl]}"
        color = 'green' if true_lbl == pred_lbl else 'red'
        ax.set_title(title, color=color, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # Hyperparameters
    n_components = 80  # Increased PCA components to capture more variance
    n_samples = 20000
    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-3
    hidden_dims = [64]  # Simplified: single small hidden layer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"PCA n_components: {n_components}")
    print(f"Total samples: {n_samples}")

    # Load and preprocess data (train/val/test)
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_cifar10(
        n_samples=n_samples, test_size=0.2, val_size=0.2
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

    # Visualize some test images with true vs predicted labels
    try:
        dataset_all = CIFAR10(root="data", train=True, download=False)
        classes = dataset_all.classes
    except Exception:
        # fallback labels
        classes = [str(i) for i in range(10)]

    # X_test (flattened) and X_test_pca are available from above
    visualize_predictions(model, X_test, X_test_pca, y_test, device, classes, num_images=16)


if __name__ == "__main__":
    main()
