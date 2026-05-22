import numpy as np
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Use CIFAR-10 as the dataset via torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

# Load CIFAR-10 (train set) and flatten images to vectors
dataset = CIFAR10(root="data", train=True, download=True, transform=ToTensor())

# Convert to numpy array of shape (n_samples, n_features) where n_features = 3*32*32
X_list = []
for img, _ in dataset:
    # img is a torch.Tensor with shape (C, H, W)
    arr = img.numpy()
    arr = np.transpose(arr, (1, 2, 0)).reshape(-1)
    X_list.append(arr)
X = np.vstack(X_list)

# For speed, optionally subsample (keep same semantics as previous synthetic data)
n_samples = min(10000, X.shape[0])
rng = np.random.RandomState(42)
idx = rng.choice(X.shape[0], n_samples, replace=False)
X = X[idx]

# Benchmark different numbers of components
results = []

# Determine valid range for n_components: cannot exceed min(n_samples, n_features)
n_features = X.shape[1]
max_components = min(X.shape[0], n_features)

# Build a sensible list of component counts (mix of absolute values and percentages)
candidate_abs = [10, 50, 100, 200, 500, 1000, 1500]
candidate_perc = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
components_set = set()
for c in candidate_abs:
    if c <= max_components:
        components_set.add(c)
for p in candidate_perc:
    val = max(1, int(max_components * p))
    if val <= max_components:
        components_set.add(val)

components_list = sorted(components_set)

print(f"n_samples={X.shape[0]}, n_features={n_features}, max_components={max_components}")

for n_components in components_list:
    pca = PCA(n_components=n_components)

    start = time.perf_counter()
    X_reduced = pca.fit_transform(X)
    elapsed = time.perf_counter() - start

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"n_components={n_components:4d} | time={elapsed:.3f}s | variance explained={variance_explained:.3f}")

    # Record results for plotting
    results.append((n_components, elapsed, variance_explained))

# Convert results to numpy array for easier indexing
results = np.array(results)

# Plotting
plt.figure(figsize=(12, 6))

# Subplot 1: Time vs. Number of Components
plt.subplot(1, 2, 1)
plt.plot(results[:, 0], results[:, 1], marker='o')
plt.title("PCA: Time vs. Number of Components")
plt.xlabel("Number of Components")
plt.ylabel("Time (seconds)")
plt.grid()

# Subplot 2: Variance Explained vs. Number of Components
plt.subplot(1, 2, 2)
plt.plot(results[:, 0], results[:, 2], marker='o', color='orange')
plt.title("PCA: Variance Explained vs. Number of Components")
plt.xlabel("Number of Components")
plt.ylabel("Variance Explained")
plt.grid()

plt.tight_layout()
plt.show()