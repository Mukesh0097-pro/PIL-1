"""
PIL Utilities: Numerical Linear Algebra Utilities for Pseudoinverse Learning.

This module provides numerically stable matrix operations for the Bi-PIL architecture.
All operations use torch.linalg for precision and stability.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import structlog

logger = structlog.get_logger(__name__)


def safe_inverse(
    matrix: torch.Tensor,
    reg_lambda: float = 1e-5,
    use_cholesky: bool = True,
) -> torch.Tensor:
    """
    Safe matrix inversion with regularization and fallback strategies.

    Computes (M + λI)^{-1} with numerical stability guarantees.

    Args:
        matrix: Square matrix to invert (N, N)
        reg_lambda: Ridge regression regularization parameter
        use_cholesky: If True, attempt Cholesky decomposition first

    Returns:
        Inverted matrix (N, N)
    """
    n = matrix.shape[0]
    device = matrix.device
    dtype = matrix.dtype

    # Add regularization: M + λI
    regularized = matrix + reg_lambda * torch.eye(n, device=device, dtype=dtype)

    # Check condition number
    cond = condition_number(regularized)
    if cond > 1e10:
        logger.warning(
            "matrix_ill_conditioned",
            condition_number=cond.item(),
            fallback="pseudoinverse",
        )
        return torch.linalg.pinv(regularized)

    if use_cholesky:
        try:
            # Cholesky is faster and more stable for positive definite matrices
            L = torch.linalg.cholesky(regularized)
            # Solve L L^T X = I
            identity = torch.eye(n, device=device, dtype=dtype)
            return torch.cholesky_solve(identity, L)
        except RuntimeError:
            logger.debug("cholesky_failed", fallback="lstsq")

    # Fallback to least squares solver
    try:
        identity = torch.eye(n, device=device, dtype=dtype)
        result = torch.linalg.lstsq(regularized, identity).solution
        return result
    except RuntimeError as e:
        logger.warning("lstsq_failed", error=str(e), fallback="pinv")
        return torch.linalg.pinv(regularized)


def condition_number(matrix: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Compute the condition number of a matrix.

    κ(A) = σ_max / σ_min

    Args:
        matrix: Input matrix
        p: Norm type (default 2-norm using SVD)

    Returns:
        Condition number (scalar tensor)
    """
    try:
        return torch.linalg.cond(matrix, p)
    except RuntimeError:
        # Fallback: compute via SVD
        s = torch.linalg.svdvals(matrix)
        return s[0] / (s[-1] + 1e-12)


def ridge_solve(
    H: torch.Tensor,
    Y: torch.Tensor,
    reg_lambda: float = 1e-5,
    use_svd_fallback: bool = True,
) -> torch.Tensor:
    """
    Solve the ridge regression problem: W = (H^T H + λI)^{-1} H^T Y

    This is the core PIL weight solving operation.

    Args:
        H: Feature matrix (N, D_hidden) - hidden activations
        Y: Target matrix (N, D_out) - targets
        reg_lambda: Regularization parameter
        use_svd_fallback: Use SVD-based pseudoinverse as fallback

    Returns:
        Weight matrix W (D_hidden, D_out)
    """
    N, D_hidden = H.shape
    device = H.device
    dtype = H.dtype

    # Compute H^T H (D_hidden, D_hidden)
    HtH = H.T @ H

    # Compute H^T Y (D_hidden, D_out)
    HtY = H.T @ Y

    # Check condition number before solving
    cond = condition_number(
        HtH + reg_lambda * torch.eye(D_hidden, device=device, dtype=dtype)
    )

    if cond > 1e10:
        logger.warning(
            "ridge_solve_ill_conditioned",
            condition_number=cond.item(),
            method="pseudoinverse",
        )
        if use_svd_fallback:
            # Use pseudoinverse directly: W = H^+ Y
            H_pinv = torch.linalg.pinv(H)
            return H_pinv @ Y

    # Standard solution: W = (H^T H + λI)^{-1} H^T Y
    try:
        # Use torch.linalg.solve for numerical stability
        reg_matrix = HtH + reg_lambda * torch.eye(D_hidden, device=device, dtype=dtype)
        W = torch.linalg.solve(reg_matrix, HtY)
        return W
    except RuntimeError as e:
        logger.warning("solve_failed", error=str(e), fallback="lstsq")
        reg_matrix = HtH + reg_lambda * torch.eye(D_hidden, device=device, dtype=dtype)
        return torch.linalg.lstsq(reg_matrix, HtY).solution


def low_rank_ridge_solve(
    H: torch.Tensor,
    Y: torch.Tensor,
    reg_lambda: float = 1e-5,
    rank: Optional[int] = None,
) -> torch.Tensor:
    """
    Low-rank approximation for ridge regression using Randomized SVD.

    For large N (tokens > 10000), use this instead of standard ridge_solve
    to reduce O(N^3) to O(N * rank^2).

    Args:
        H: Feature matrix (N, D_hidden)
        Y: Target matrix (N, D_out)
        reg_lambda: Regularization parameter
        rank: Rank for low-rank approximation (default: min(100, D_hidden//2))

    Returns:
        Weight matrix W (D_hidden, D_out)
    """
    N, D_hidden = H.shape
    device = H.device
    dtype = H.dtype

    if rank is None:
        rank = min(100, D_hidden // 2, N // 2)
    rank = max(1, rank)

    # Truncated SVD of H
    try:
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)

        # Keep top-k singular values
        k = min(rank, len(S))
        U_k = U[:, :k]  # (N, k)
        S_k = S[:k]  # (k,)
        Vh_k = Vh[:k, :]  # (k, D_hidden)

        # Regularized inverse of singular values
        # (S^2 + λ)^{-1} * S
        S_reg = S_k / (S_k**2 + reg_lambda)

        # W = V * diag(S_reg) * U^T * Y
        UY = U_k.T @ Y  # (k, D_out)
        W = Vh_k.T @ (S_reg.unsqueeze(1) * UY)  # (D_hidden, D_out)

        return W

    except RuntimeError as e:
        logger.warning("low_rank_svd_failed", error=str(e), fallback="standard")
        return ridge_solve(H, Y, reg_lambda)


def sherman_morrison_update(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    h_new: torch.Tensor,
    y_new: torch.Tensor,
    reg_lambda: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Incremental weight update using Sherman-Morrison formula.

    For online learning, update weights one sample at a time without
    full matrix inversion.

    (A + uv^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)

    Args:
        W: Current weight matrix (D_hidden, D_out)
        H_inv: Current (H^T H + λI)^{-1} matrix (D_hidden, D_hidden)
        h_new: New hidden activation (D_hidden,)
        y_new: New target (D_out,)
        reg_lambda: Regularization parameter

    Returns:
        Tuple of (updated W, updated H_inv)
    """
    h = h_new.unsqueeze(1)  # (D_hidden, 1)

    # Sherman-Morrison update for H_inv
    # H_inv_new = H_inv - (H_inv @ h @ h^T @ H_inv) / (1 + h^T @ H_inv @ h)
    H_inv_h = H_inv @ h  # (D_hidden, 1)
    denom = 1.0 + (h.T @ H_inv_h).squeeze()

    if denom.abs() < 1e-10:
        logger.warning("sherman_morrison_singular", denom=denom.item())
        return W, H_inv

    H_inv_new = H_inv - (H_inv_h @ H_inv_h.T) / denom

    # Update weights: W_new = H_inv_new @ H^T @ Y_new
    # Simplified: W_new = W + H_inv_new @ h @ (y_new - h^T @ W)
    residual = y_new - (h.T @ W).squeeze()
    W_new = W + H_inv_new @ h @ residual.unsqueeze(0)

    return W_new, H_inv_new


def orthogonal_init(
    shape: Tuple[int, int],
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Initialize a random orthogonal matrix using SVD.

    For the fixed expansion weights W_random in PIL.

    Args:
        shape: (rows, cols) shape of the matrix
        device: Target device
        dtype: Data type
        seed: Random seed for reproducibility

    Returns:
        Orthogonal matrix (rows, cols)
    """
    if seed is not None:
        torch.manual_seed(seed)

    rows, cols = shape

    # Generate random matrix
    M = torch.randn(rows, cols, device=device, dtype=dtype)

    # Orthogonalize via SVD
    if rows >= cols:
        U, _, _ = torch.linalg.svd(M, full_matrices=False)
        return U[:, :cols]
    else:
        _, _, Vh = torch.linalg.svd(M, full_matrices=False)
        return Vh[:rows, :]


class NumericalMonitor:
    """
    Monitor numerical stability during PIL training.

    Tracks condition numbers, NaN occurrences, and other stability metrics.
    """

    def __init__(self, warn_threshold: float = 1e8, fail_threshold: float = 1e12):
        self.warn_threshold = warn_threshold
        self.fail_threshold = fail_threshold
        self.history = {
            "condition_numbers": [],
            "nan_count": 0,
            "inf_count": 0,
            "warnings": [],
        }

    def check_matrix(self, matrix: torch.Tensor, name: str = "matrix") -> bool:
        """
        Check a matrix for numerical issues.

        Args:
            matrix: Matrix to check
            name: Name for logging

        Returns:
            True if matrix is stable, False otherwise
        """
        # Check for NaN
        if torch.isnan(matrix).any():
            self.history["nan_count"] += 1
            logger.error(f"nan_detected_in_{name}")
            return False

        # Check for Inf
        if torch.isinf(matrix).any():
            self.history["inf_count"] += 1
            logger.error(f"inf_detected_in_{name}")
            return False

        # Check condition number for square matrices
        if matrix.shape[0] == matrix.shape[1]:
            cond = condition_number(matrix)
            self.history["condition_numbers"].append(cond.item())

            if cond > self.fail_threshold:
                logger.error(f"{name}_condition_critical", condition_number=cond.item())
                return False
            elif cond > self.warn_threshold:
                self.history["warnings"].append(f"{name}: κ={cond.item():.2e}")
                logger.warning(
                    f"{name}_condition_warning", condition_number=cond.item()
                )

        return True

    def get_summary(self) -> dict:
        """Get summary of numerical stability."""
        cond_nums = self.history["condition_numbers"]
        return {
            "nan_count": self.history["nan_count"],
            "inf_count": self.history["inf_count"],
            "warning_count": len(self.history["warnings"]),
            "max_condition_number": max(cond_nums) if cond_nums else 0,
            "mean_condition_number": sum(cond_nums) / len(cond_nums)
            if cond_nums
            else 0,
        }

    def reset(self):
        """Reset monitoring history."""
        self.history = {
            "condition_numbers": [],
            "nan_count": 0,
            "inf_count": 0,
            "warnings": [],
        }
