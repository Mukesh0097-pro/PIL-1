import numpy as np
import structlog
from typing import Optional

logger = structlog.get_logger(__name__)


class PILVAE:
    """
    PIL-VAE: Pseudoinverse Learning Variational Autoencoder.
    Gradient-free VAE implementation with numerical stability improvements.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        alpha: float = 1e-3,
    ):
        """
        Initialize PIL-VAE.

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
        self.W1: Optional[np.ndarray] = None  # Encoder: Random Orthogonal (h x d)
        self.W_proj: Optional[np.ndarray] = None  # Encoder: PPCA Projection (h x q)
        self.mu: Optional[np.ndarray] = None  # Encoder: PPCA Mean (h x 1)
        self.W4: Optional[np.ndarray] = None  # Decoder: Latent -> Hidden (h x q)
        self.W6: Optional[np.ndarray] = None  # Decoder: Hidden -> Input (d x h)

        # State tracking
        self._is_fitted = False

        self._init_random_weights()

    def _init_random_weights(self):
        """Initialize encoder weights with random orthogonal matrix."""
        rng = np.random.default_rng(seed=42)  # Reproducibility
        M = rng.standard_normal((self.h, self.d))
        # Orthogonalize via SVD
        U, _, _ = np.linalg.svd(M.T, full_matrices=False)
        self.W1 = U.T  # (h, d)

    def leaky_relu(self, x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """Leaky ReLU activation with numerical stability."""
        return np.maximum(alpha * x, x)

    def _safe_inverse(self, matrix: np.ndarray, reg_param: float) -> np.ndarray:
        """
        Safe matrix inversion with fallback to pseudoinverse.

        Args:
            matrix: Square matrix to invert
            reg_param: Regularization parameter

        Returns:
            Inverted matrix
        """
        n = matrix.shape[0]
        regularized = matrix + reg_param * np.eye(n)

        try:
            # Check condition number first
            cond = np.linalg.cond(regularized)
            if cond > 1e10:
                logger.warning("matrix_ill_conditioned", condition_number=cond)
                return np.linalg.pinv(regularized)

            return np.linalg.inv(regularized)
        except np.linalg.LinAlgError as e:
            logger.warning("matrix_inversion_failed", error=str(e), fallback="pinv")
            return np.linalg.pinv(regularized)

    def fit(self, X_emb: np.ndarray) -> None:
        """
        Fit the VAE on a batch of embeddings.

        Args:
            X_emb: (N, d) numpy array of embeddings.
        """
        if len(X_emb) == 0:
            logger.warning("fit_called_with_empty_data")
            return

        # Validate input
        if X_emb.shape[1] != self.d:
            raise ValueError(f"Expected embedding dim {self.d}, got {X_emb.shape[1]}")

        # Transpose to (d, N) for math consistency
        X = X_emb.T.astype(np.float64)  # Use float64 for numerical stability
        N = X.shape[1]

        # --- 1. Reductive Subsystem ---

        # Hidden Activations: H1 = f(W1 X)
        H1 = self.leaky_relu(self.W1.astype(np.float64) @ X)

        # PPCA on H1^T - project H1 (h-dim) down to Z (q-dim)
        self.mu = np.mean(H1, axis=1, keepdims=True)
        H_centered = H1 - self.mu

        # SVD on centered data for PCA
        U, S, _ = np.linalg.svd(H_centered, full_matrices=False)

        # Projection matrix W (h x q)
        # Handle case when N < q (fewer samples than latent dims)
        actual_q = min(self.q, U.shape[1])
        self.W_proj = np.zeros((self.h, self.q), dtype=np.float64)
        self.W_proj[:, :actual_q] = U[:, :actual_q]

        # Latent Codes: Z = W^T (H1 - mu)
        Z = self.W_proj.T @ H_centered

        # --- 2. Generative Subsystem (Decoder) ---

        # Learn W4: Map Z -> H1
        # W4 = H1 Z^T (ZZ^T + lambda I)^-1
        ZZT = Z @ Z.T
        inv_term = self._safe_inverse(ZZT, self.alpha)
        self.W4 = (H1 @ Z.T) @ inv_term

        # Learn W6: Map H1 -> X
        # W6 = X H1^T (H1 H1^T + lambda I)^-1
        H1H1T = H1 @ H1.T
        inv_term_h = self._safe_inverse(H1H1T, self.alpha)
        self.W6 = (X @ H1.T) @ inv_term_h

        # Convert back to float32 for memory efficiency
        self.W4 = self.W4.astype(np.float32)
        self.W6 = self.W6.astype(np.float32)
        self.mu = self.mu.astype(np.float32)
        self.W_proj = self.W_proj.astype(np.float32)

        self._is_fitted = True
        logger.debug("vae_fitted", num_samples=N, latent_dim=self.q)

    def encode(self, x_vec: np.ndarray) -> np.ndarray:
        """
        Map single vector (d,) to latent (q,).

        Args:
            x_vec: Input vector of dimension d

        Returns:
            Latent vector of dimension q
        """
        if not self._is_fitted:
            # Return zero vector if not fitted
            return np.zeros(self.q, dtype=np.float32)

        x = x_vec.reshape(-1, 1).astype(np.float32)
        h1 = self.leaky_relu(self.W1 @ x)
        z = self.W_proj.T @ (h1 - self.mu)
        return z.flatten()

    def decode(self, z_vec: np.ndarray) -> np.ndarray:
        """
        Map latent (q,) to reconstruction (d,).

        Args:
            z_vec: Latent vector of dimension q

        Returns:
            Reconstructed vector of dimension d
        """
        if not self._is_fitted:
            return np.zeros(self.d, dtype=np.float32)

        z = z_vec.reshape(-1, 1).astype(np.float32)
        h_rec = self.W4 @ z
        x_rec = self.W6 @ h_rec
        return x_rec.flatten()

    def reconstruct(self, x_vec: np.ndarray) -> np.ndarray:
        """
        Encode and decode (full reconstruction).

        Args:
            x_vec: Input vector

        Returns:
            Reconstructed vector
        """
        z = self.encode(x_vec)
        return self.decode(z)

    def generate(self, n_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample z ~ N(0, I) and decode.

        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            Array of generated samples (n_samples, d)
        """
        if not self._is_fitted:
            logger.warning("generate_called_before_fit")
            return np.zeros((n_samples, self.d), dtype=np.float32)

        rng = np.random.default_rng(seed=seed)
        z_samples = rng.standard_normal((self.q, n_samples))

        generated = []
        for i in range(n_samples):
            z = z_samples[:, i]
            x_gen = self.decode(z)
            generated.append(x_gen)
        return np.array(generated, dtype=np.float32)

    def get_reconstruction_error(self, X_emb: np.ndarray) -> float:
        """
        Calculate mean reconstruction error.

        Args:
            X_emb: (N, d) array of embeddings

        Returns:
            Mean squared reconstruction error
        """
        if not self._is_fitted or len(X_emb) == 0:
            return float("inf")

        errors = []
        for x in X_emb:
            x_rec = self.reconstruct(x)
            error = np.mean((x - x_rec) ** 2)
            errors.append(error)
        return np.mean(errors)

    @property
    def is_fitted(self) -> bool:
        """Check if VAE has been fitted."""
        return self._is_fitted
