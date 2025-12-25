"""
Bi-PIL Layer: Bidirectional Pseudoinverse Learning Layer.

This module implements the core FFN replacement for the Attention-PIL Hybrid architecture.
Training uses algebraic weight solving via pseudoinverse, NOT backpropagation.

Reference:
- "Bi-PIL: Bidirectional Gradient-Free Learning Scheme"
- "SONG: Synergetic Learning System Based on Swarm of Non-Gradient Learners"
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
import structlog

from app.core.pil_utils import (
    ridge_solve,
    low_rank_ridge_solve,
    orthogonal_init,
    NumericalMonitor,
    condition_number,
)

logger = structlog.get_logger(__name__)


class PILLayer(nn.Module):
    """
    Single-direction Pseudoinverse Learning Layer.

    Implements: y = activation(x @ W_random) @ W_out

    W_random: Fixed random orthogonal expansion (requires_grad=False)
    W_out: Solved via pseudoinverse (requires_grad=False)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "gelu",
        reg_lambda: float = 1e-5,
        seed: Optional[int] = None,
    ):
        """
        Initialize PIL Layer.

        Args:
            input_dim: Input dimension (D)
            hidden_dim: Hidden expansion dimension (H) - typically 4x input_dim
            output_dim: Output dimension (D_out)
            activation: Activation function ("gelu", "relu", "leaky_relu", "silu")
            reg_lambda: Ridge regularization parameter (λ)
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reg_lambda = reg_lambda

        # Fixed random expansion weights (NOT trained via backprop)
        self.register_buffer(
            "W_random", orthogonal_init((input_dim, hidden_dim), seed=seed)
        )

        # Output weights (solved via pseudoinverse, NOT trained via backprop)
        # Using register_buffer instead of nn.Parameter since we don't want gradients
        self.register_buffer("W_out", torch.zeros(hidden_dim, output_dim))

        # Bias (optional, also solved)
        self.register_buffer("bias", torch.zeros(output_dim))

        # Activation function
        self.activation = self._get_activation(activation)

        # State tracking
        self._is_fitted = False
        self.monitor = NumericalMonitor()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.GELU())

    def _expand(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature expansion: H = activation(X @ W_random)

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Hidden activations (..., hidden_dim)
        """
        return self.activation(x @ self.W_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = H @ W_out + bias

        Args:
            x: Input tensor (B, S, D) or (N, D)

        Returns:
            Output tensor (B, S, D_out) or (N, D_out)
        """
        h = self._expand(x)
        return h @ self.W_out + self.bias

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        use_low_rank: bool = False,
        rank: Optional[int] = None,
    ) -> dict:
        """
        Solve for W_out using pseudoinverse.

        W_out = (H^T H + λI)^{-1} H^T Y

        Args:
            x: Input tensor (N, D) - flattened batch
            target: Target tensor (N, D_out)
            use_low_rank: Use low-rank SVD approximation for large N
            rank: Rank for low-rank approximation

        Returns:
            Dictionary with fit statistics
        """
        # Ensure 2D tensors
        if x.dim() > 2:
            orig_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            target = target.reshape(-1, target.shape[-1])

        # Feature expansion
        H = self._expand(x)  # (N, hidden_dim)

        # Check numerical stability
        if not self.monitor.check_matrix(H, "hidden_activations"):
            logger.error("fit_aborted_numerical_instability")
            return {"success": False, "reason": "numerical_instability"}

        # Solve for weights
        N = H.shape[0]

        if use_low_rank or N > 10000:
            W_new = low_rank_ridge_solve(H, target, self.reg_lambda, rank)
            method = "low_rank_svd"
        else:
            W_new = ridge_solve(H, target, self.reg_lambda)
            method = "ridge_solve"

        # Check solution stability
        if not self.monitor.check_matrix(W_new, "W_out"):
            logger.error("fit_produced_unstable_weights")
            return {"success": False, "reason": "unstable_weights"}

        # Update weights in-place
        self.W_out.copy_(W_new)

        # Compute bias as mean residual (optional)
        residual = target - (H @ self.W_out)
        self.bias.copy_(residual.mean(dim=0))

        self._is_fitted = True

        # Compute fit statistics
        y_pred = H @ self.W_out + self.bias
        mse = ((target - y_pred) ** 2).mean().item()

        logger.debug(
            "pil_layer_fitted",
            n_samples=N,
            method=method,
            mse=mse,
        )

        return {
            "success": True,
            "method": method,
            "n_samples": N,
            "mse": mse,
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class BiPILLayer(nn.Module):
    """
    Bidirectional Pseudoinverse Learning Layer (Bi-PIL).

    Implements two parallel expansion flows:
    - Forward: H_fwd = σ(X @ W_fwd)
    - Backward: H_bwd = σ(X @ W_bwd)
    - Fusion: H = concat(H_fwd, H_bwd) or H = H_fwd + H_bwd

    This replaces the standard FFN in Transformer blocks.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: int = 4,
        activation: str = "gelu",
        fusion: Literal["concat", "add", "gate"] = "concat",
        reg_lambda: float = 1e-5,
        seed: Optional[int] = None,
    ):
        """
        Initialize Bi-PIL Layer.

        Args:
            dim: Model dimension (D)
            expansion_factor: Hidden dim = dim * expansion_factor
            activation: Activation function
            fusion: How to combine forward/backward features
            reg_lambda: Ridge regularization parameter
            seed: Random seed
        """
        super().__init__()

        self.dim = dim
        self.hidden_dim = dim * expansion_factor
        self.fusion = fusion
        self.reg_lambda = reg_lambda

        # Determine output hidden dim based on fusion
        if fusion == "concat":
            self.fused_dim = self.hidden_dim * 2
        else:
            self.fused_dim = self.hidden_dim

        # Forward expansion: W_fwd is FIXED (requires_grad=False)
        self.register_buffer(
            "W_fwd", orthogonal_init((dim, self.hidden_dim), seed=seed)
        )

        # Backward expansion: W_bwd is FIXED (requires_grad=False)
        self.register_buffer(
            "W_bwd",
            orthogonal_init((dim, self.hidden_dim), seed=seed + 1 if seed else None),
        )

        # Output projection: W_out is SOLVED via pseudoinverse
        self.register_buffer("W_out", torch.zeros(self.fused_dim, dim))

        # Optional gating for fusion
        if fusion == "gate":
            self.register_buffer("gate_weights", torch.ones(self.hidden_dim) * 0.5)

        # Bias
        self.register_buffer("bias", torch.zeros(dim))

        # Activation
        self.activation = self._get_activation(activation)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(dim)

        # State
        self._is_fitted = False
        self.monitor = NumericalMonitor()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.GELU())

    def _expand_bidirectional(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional feature expansion.

        Args:
            x: Input (..., dim)

        Returns:
            Fused hidden features (..., fused_dim)
        """
        # Forward expansion: H_fwd = σ(X @ W_fwd)
        H_fwd = self.activation(x @ self.W_fwd)

        # Backward expansion: H_bwd = σ(X @ W_bwd)
        H_bwd = self.activation(x @ self.W_bwd)

        # Fusion
        if self.fusion == "concat":
            return torch.cat([H_fwd, H_bwd], dim=-1)
        elif self.fusion == "add":
            return H_fwd + H_bwd
        elif self.fusion == "gate":
            # Learned gating (but gate_weights are also solved, not gradient-trained)
            gate = torch.sigmoid(self.gate_weights)
            return gate * H_fwd + (1 - gate) * H_bwd
        else:
            return H_fwd + H_bwd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor (B, S, D)

        Returns:
            Output tensor (B, S, D)
        """
        # Store residual
        residual = x

        # Bidirectional expansion
        H = self._expand_bidirectional(x)

        # Output projection
        out = H @ self.W_out + self.bias

        # Residual connection + LayerNorm
        out = self.layer_norm(out + residual)

        return out

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_low_rank: bool = False,
        rank: Optional[int] = None,
    ) -> dict:
        """
        Solve for W_out using pseudoinverse.

        If target is None, uses residual learning: target = x (identity mapping).

        Args:
            x: Input tensor (B, S, D) or (N, D)
            target: Target tensor (same shape as x). If None, learns identity.
            use_low_rank: Use low-rank approximation
            rank: Rank for approximation

        Returns:
            Fit statistics dictionary
        """
        # Flatten to 2D
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)

        # Default target is identity (residual learning)
        if target is None:
            target_flat = x_flat.clone()
        else:
            target_flat = target.reshape(-1, self.dim)

        # Bidirectional expansion
        H = self._expand_bidirectional(x_flat)  # (N, fused_dim)

        # Check numerical stability
        if not self.monitor.check_matrix(H, "bi_hidden"):
            return {"success": False, "reason": "numerical_instability"}

        N = H.shape[0]

        # Solve ridge regression: W_out = (H^T H + λI)^{-1} H^T Y
        if use_low_rank or N > 10000:
            W_new = low_rank_ridge_solve(H, target_flat, self.reg_lambda, rank)
            method = "low_rank_svd"
        else:
            W_new = ridge_solve(H, target_flat, self.reg_lambda)
            method = "ridge_solve"

        # Check solution
        if not self.monitor.check_matrix(W_new, "bi_W_out"):
            return {"success": False, "reason": "unstable_weights"}

        # Update weights
        self.W_out.copy_(W_new)

        # Update bias
        residual = target_flat - (H @ self.W_out)
        self.bias.copy_(residual.mean(dim=0))

        self._is_fitted = True

        # Statistics
        y_pred = H @ self.W_out + self.bias
        mse = ((target_flat - y_pred) ** 2).mean().item()

        cond = condition_number(H.T @ H).item()

        logger.debug(
            "bipil_layer_fitted",
            n_samples=N,
            method=method,
            mse=mse,
            condition_number=cond,
        )

        return {
            "success": True,
            "method": method,
            "n_samples": N,
            "mse": mse,
            "condition_number": cond,
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_effective_rank(self) -> int:
        """Compute effective rank of the learned transformation."""
        if not self._is_fitted:
            return 0
        s = torch.linalg.svdvals(self.W_out)
        threshold = s.max() * 1e-5
        return (s > threshold).sum().item()


class SwarmPIL(nn.Module):
    """
    Swarm of Non-Gradient Learners (SONG Implementation).

    Uses a ModuleList of small PIL learners and averages their outputs
    for improved robustness and diversity.

    Reference: "SONG: Synergetic Learning System Based on Swarm of Non-Gradient Learners"
    """

    def __init__(
        self,
        dim: int,
        n_learners: int = 4,
        expansion_factor: int = 2,
        reg_lambda: float = 1e-5,
        seed: Optional[int] = None,
    ):
        """
        Initialize Swarm PIL.

        Args:
            dim: Model dimension
            n_learners: Number of parallel learners in the swarm
            expansion_factor: Expansion factor per learner (smaller than BiPIL)
            reg_lambda: Regularization parameter
            seed: Random seed
        """
        super().__init__()

        self.dim = dim
        self.n_learners = n_learners

        # Create swarm of small PIL learners
        self.learners = nn.ModuleList(
            [
                PILLayer(
                    input_dim=dim,
                    hidden_dim=dim * expansion_factor,
                    output_dim=dim,
                    reg_lambda=reg_lambda,
                    seed=seed + i if seed else None,
                )
                for i in range(n_learners)
            ]
        )

        # Optional learner weights (for weighted averaging)
        self.register_buffer("learner_weights", torch.ones(n_learners) / n_learners)

        # Layer norm
        self.layer_norm = nn.LayerNorm(dim)

        self._is_fitted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: weighted average of all learners.

        Args:
            x: Input (B, S, D)

        Returns:
            Output (B, S, D)
        """
        residual = x

        # Collect outputs from all learners
        outputs = []
        for i, learner in enumerate(self.learners):
            out = learner(x)
            outputs.append(self.learner_weights[i] * out)

        # Average
        combined = sum(outputs)

        # Residual + LayerNorm
        return self.layer_norm(combined + residual)

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Fit all learners in the swarm.

        Args:
            x: Input tensor
            target: Target tensor

        Returns:
            Combined fit statistics
        """
        if target is None:
            target = x

        results = []
        for i, learner in enumerate(self.learners):
            # Each learner gets the same input/target
            result = learner.fit(
                x.reshape(-1, self.dim), target.reshape(-1, self.dim), **kwargs
            )
            results.append(result)

        self._is_fitted = all(r.get("success", False) for r in results)

        # Aggregate statistics
        mses = [r.get("mse", float("inf")) for r in results]

        return {
            "success": self._is_fitted,
            "n_learners": self.n_learners,
            "mean_mse": sum(mses) / len(mses),
            "min_mse": min(mses),
            "max_mse": max(mses),
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
