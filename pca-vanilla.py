import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

# Generate synthetic data
n_samples, n_features = 10000, 500
X, _ = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)

# Benchmark different numbers of components
for n_components in [10, 50, 100, 200]:
    pca = PCA(n_components=n_components)
    
    start = time.perf_counter()
    X_reduced = pca.fit_transform(X)
    elapsed = time.perf_counter() - start
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"n_components={n_components:3d} | "
          f"time={elapsed:.3f}s | "
          f"variance explained={variance_explained:.3f}")