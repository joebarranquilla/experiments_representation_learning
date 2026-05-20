import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate synthetic data
n_samples, n_features = 10000, 500
X, _ = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)

# Benchmark different numbers of components
results = []
for n_components in [10, 50, 100, 200, 500]:
    pca = PCA(n_components=n_components)
    
    start = time.perf_counter()
    X_reduced = pca.fit_transform(X)
    elapsed = time.perf_counter() - start
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"n_components={n_components:3d} | "
          f"time={elapsed:.3f}s | "
          f"variance explained={variance_explained:.3f}")
#Now graph the results
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