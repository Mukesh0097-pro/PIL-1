"""
Core PIL Components for Project Emergent-1.

This package provides:
    - pil_utils: Numerical linear algebra utilities
    - bipil_layer: Bi-directional Pseudoinverse Learning layers
    - attention_pil: Hybrid Attention-PIL Transformer blocks and model
    - pil_vae: PIL-based Variational Autoencoder
"""

from app.core.pil_utils import (
    safe_inverse,
    condition_number,
    ridge_solve,
    low_rank_ridge_solve,
    orthogonal_init,
    sherman_morrison_update,
    NumericalMonitor,
)

from app.core.bipil_layer import (
    PILLayer,
    BiPILLayer,
    SwarmPIL,
)

from app.core.attention_pil import (
    MultiHeadAttention,
    AttentionPILBlock,
    AttentionPILModel,
    PILTrainer,
)

from app.core.pil_vae import PILVAE

__all__ = [
    # Utilities
    "safe_inverse",
    "condition_number",
    "ridge_solve",
    "low_rank_ridge_solve",
    "orthogonal_init",
    "sherman_morrison_update",
    "NumericalMonitor",
    # PIL Layers
    "PILLayer",
    "BiPILLayer",
    "SwarmPIL",
    # Attention-PIL
    "MultiHeadAttention",
    "AttentionPILBlock",
    "AttentionPILModel",
    "PILTrainer",
    # VAE
    "PILVAE",
]
