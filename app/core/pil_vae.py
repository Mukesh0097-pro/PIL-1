import numpy as np


class PILVAE:
    def __init__(self, input_dim, latent_dim=16, hidden_dim=64, alpha=1e-3):
        """
        PIL-VAE: Pseudoinverse Learning Variational Autoencoder.

        Args:
            input_dim: Dimension of input embeddings (d).
            latent_dim: Dimension of latent space (q).
            hidden_dim: Dimension of hidden layer (h).
            alpha: Ridge regression regularization parameter (lambda).
        """
        self.d = input_dim
        self.q = latent_dim
        self.h = hidden_dim
        self.alpha = alpha

        # Weights
        self.W1 = None  # Encoder: Random Orthogonal (h x d)
        self.W_proj = None  # Encoder: PPCA Projection (h x q)
        self.mu = None  # Encoder: PPCA Mean (h x 1)
        self.W4 = None  # Decoder: Latent -> Hidden (h x q)
        self.W6 = None  # Decoder: Hidden -> Input (d x h)

        self._init_random_weights()

    def _init_random_weights(self):
        # Generate random matrix
        rng = np.random.default_rng()
        M = rng.standard_normal((self.h, self.d))
        # Orthogonalize
        U, _, _ = np.linalg.svd(M.T, full_matrices=False)
        self.W1 = U.T  # (h, d)

    def leaky_relu(self, x, alpha=0.1):
        return np.maximum(alpha * x, x)

    def fit(self, X_emb):
        """
        Fit the VAE on a batch of embeddings.
        X_emb: (N, d) numpy array of embeddings.
        """
        if len(X_emb) == 0:
            return

        # Transpose to (d, N) for math consistency with prompt
        X = X_emb.T
        N = X.shape[1]

        # --- 1. Reductive Subsystem ---

        # Hidden Activations: H1 = f(W1 X)
        # W1: (h, d), X: (d, N) -> H1: (h, N)
        H1 = self.leaky_relu(self.W1 @ X)

        # PPCA on H1^T
        # We want to project H1 (h-dim) down to Z (q-dim)
        # Compute mean
        self.mu = np.mean(H1, axis=1, keepdims=True)  # (h, 1)
        H_centered = H1 - self.mu

        # SVD on centered data for PCA
        # H_centered is (h, N). Covariance is (h, h).
        # U: (h, h), S: (h,), Vt: (N, N)
        U, S, _ = np.linalg.svd(H_centered, full_matrices=False)

        # Projection matrix W (h x q)
        self.W_proj = U[:, : self.q]

        # Latent Codes: Z = W^T (H1 - mu)
        # (q, h) @ (h, N) -> (q, N)
        Z = self.W_proj.T @ H_centered

        # --- 2. Generative Subsystem (Decoder) ---

        # Learn W4: Map Z -> H1
        # W4 = H1 Z^T (ZZ^T + lambda I)^-1
        # Z: (q, N), H1: (h, N)
        # Ridge Regression closed form
        reg = self.alpha * np.eye(self.q)
        inv_term = np.linalg.inv((Z @ Z.T) + reg)
        self.W4 = (H1 @ Z.T) @ inv_term  # (h, q)

        # Learn W6: Map H1 -> X
        # W6 = X H1^T (H1 H1^T + lambda I)^-1
        # X: (d, N), H1: (h, N)
        reg_h = self.alpha * np.eye(self.h)
        inv_term_h = np.linalg.inv((H1 @ H1.T) + reg_h)
        self.W6 = (X @ H1.T) @ inv_term_h  # (d, h)

    def encode(self, x_vec):
        """Map single vector (d,) to latent (q,)"""
        x = x_vec.reshape(-1, 1)  # (d, 1)
        h1 = self.leaky_relu(self.W1 @ x)
        z = self.W_proj.T @ (h1 - self.mu)
        return z.flatten()

    def decode(self, z_vec):
        """Map latent (q,) to reconstruction (d,)"""
        z = z_vec.reshape(-1, 1)  # (q, 1)
        h_rec = self.W4 @ z
        x_rec = self.W6 @ h_rec
        return x_rec.flatten()

    def generate(self, n_samples=1):
        """Sample z ~ N(0, I) and decode."""
        rng = np.random.default_rng()
        z_samples = rng.standard_normal((self.q, n_samples))

        generated = []
        for i in range(n_samples):
            z = z_samples[:, i]
            x_gen = self.decode(z)
            generated.append(x_gen)
        return np.array(generated)
