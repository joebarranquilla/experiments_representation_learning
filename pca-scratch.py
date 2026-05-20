import numpy as np


class PCA:
    def __init__(self, n_components=None):
        """
        Parameters
        ----------
        n_components : int or None
            Number of principal components to keep.
            If None, keep all components.
        """
        self.n_components = n_components

        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """
        Fit PCA on dataset X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        X = np.asarray(X)

        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        print("Eigenvalues before sorting:", eigenvalues)

        # 4. Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        print("Indices for sorting eigenvalues:", idx)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. Select top components
        if self.n_components is not None:
            eigenvalues = eigenvalues[:self.n_components]
            eigenvectors = eigenvectors[:, :self.n_components]

        self.components_ = eigenvectors.T
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = (
            eigenvalues / np.sum(np.linalg.eigvalsh(cov_matrix))
        )

        return self

    def transform(self, X):
        """
        Project data onto principal components.
        """
        X = np.asarray(X)
        X_centered = X - self.mean_

        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """
        Fit PCA and transform X.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Reconstruct data from PCA space.
        """
        return X_transformed @ self.components_ + self.mean_


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example dataset: 2000 samples, 5 features
    np.random.seed(42)
    X = np.random.randn(2000, 5)

    # Reduce to 2 dimensions
    pca = PCA(n_components=2)

    X_reduced = pca.fit_transform(X)

    print("Original shape:", X.shape)
    print("Reduced shape:", X_reduced.shape)

    print("\nPrincipal components:")
    print(pca.components_)

    print("\nExplained variance:")
    print(pca.explained_variance_)

    print("\nExplained variance ratio:")
    print(pca.explained_variance_ratio_)

    # Plot the reduced data
    import matplotlib.pyplot as plt

    from sklearn.manifold import TSNE

    # Compute t-SNE embedding (might take a bit of time for large datasets)
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(12, 5))

    # Left: PCA reduced data
    plt.subplot(1, 2, 1)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=8, alpha=0.7)
    plt.title("PCA: Reduced Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)

    # Right: t-SNE on original data
    plt.subplot(1, 2, 2)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=8, alpha=0.7, color="C1")
    plt.title("t-SNE: Original Data")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)

    plt.tight_layout()
    plt.show()