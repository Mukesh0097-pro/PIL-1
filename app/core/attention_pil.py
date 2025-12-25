"""
Attention-PIL Block: Hybrid Transformer Block with PIL-based FFN.

This module implements the core Transformer block that combines:
- Standard Multi-Head Attention (gradient-based or frozen)
- Bi-PIL Feed-Forward Network (gradient-free, pseudoinverse-solved)

Reference: Project Emergent-1 PRD - Attention-PIL Hybrid Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
import structlog

from app.core.bipil_layer import BiPILLayer, SwarmPIL
from app.core.pil_utils import NumericalMonitor

logger = structlog.get_logger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention.

    This component CAN use gradients (if freeze_attention=False).
    It's the only part of the hybrid architecture that may use backprop.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        freeze: bool = False,
    ):
        """
        Initialize Multi-Head Attention.

        Args:
            dim: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            freeze: If True, attention weights are frozen (random projection)
        """
        super().__init__()

        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.freeze = freeze

        # QKV projection
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)

        # Output projection
        self.proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        if freeze:
            # Freeze attention weights
            for param in self.parameters():
                param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, S, D)
            mask: Attention mask (B, S) or (B, S, S)

        Returns:
            Tuple of (output, attention_weights)
        """
        B, S, D = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, S, S)

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # (B, S) -> (B, 1, 1, S)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (B, S, S) -> (B, 1, S, S)
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = (attn_weights @ v).transpose(1, 2).reshape(B, S, D)
        out = self.proj(out)

        return out, attn_weights


class AttentionPILBlock(nn.Module):
    """
    Hybrid Transformer Block: Attention + Bi-PIL FFN.

    Architecture:
        1. Multi-Head Self-Attention (gradient-based or frozen)
        2. Add & Norm
        3. Bi-PIL FFN (gradient-free, solved via pseudoinverse)
        4. Add & Norm

    Training:
        - Attention: Standard backprop (if not frozen)
        - FFN: .fit() method with matrix inversion
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        freeze_attention: bool = False,
        ffn_type: Literal["bipil", "swarm"] = "bipil",
        reg_lambda: float = 1e-5,
        n_swarm_learners: int = 4,
    ):
        """
        Initialize Attention-PIL Block.

        Args:
            dim: Model dimension
            n_heads: Number of attention heads
            expansion_factor: FFN expansion factor
            dropout: Dropout rate
            freeze_attention: Freeze attention weights
            ffn_type: Type of PIL FFN ("bipil" or "swarm")
            reg_lambda: Ridge regression parameter for PIL
            n_swarm_learners: Number of learners if using swarm
        """
        super().__init__()

        self.dim = dim

        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
            freeze=freeze_attention,
        )

        # Attention LayerNorm
        self.attn_norm = nn.LayerNorm(dim)

        # FFN: Bi-PIL or Swarm
        if ffn_type == "swarm":
            self.ffn = SwarmPIL(
                dim=dim,
                n_learners=n_swarm_learners,
                expansion_factor=expansion_factor // 2,
                reg_lambda=reg_lambda,
            )
        else:
            self.ffn = BiPILLayer(
                dim=dim,
                expansion_factor=expansion_factor,
                reg_lambda=reg_lambda,
            )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # State tracking
        self._attn_cache = None
        self._ffn_fitted = False

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, S, D)
            mask: Attention mask
            return_attention: Return attention weights

        Returns:
            Output tensor (B, S, D), optionally attention weights
        """
        # Self-Attention + Residual + Norm
        attn_out, attn_weights = self.attention(x, mask)
        x = self.attn_norm(x + self.dropout(attn_out))

        # Cache attention output for FFN fitting
        self._attn_cache = x.detach().clone()

        # Bi-PIL FFN (includes residual + norm internally)
        x = self.ffn(x)

        if return_attention:
            return x, attn_weights
        return x

    @torch.no_grad()
    def fit_ffn(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_low_rank: bool = False,
    ) -> dict:
        """
        Fit the PIL FFN on a batch of data.

        This is the training step for the FFN - NO BACKPROP, uses pseudoinverse.

        Data Flow:
            1. Run attention to get contextualized representations
            2. Fit PIL layer: W_out = pinv(H) @ target

        Args:
            x: Input tensor (B, S, D)
            target: Target tensor. If None, uses x (identity/residual learning)
            mask: Attention mask
            use_low_rank: Use low-rank approximation for large batches

        Returns:
            Fit statistics
        """
        # Step 1: Run attention to get contextualized input
        attn_out, _ = self.attention(x, mask)
        attn_contextualized = self.attn_norm(x + attn_out)

        # Step 2: Fit PIL FFN
        # If no target, use identity mapping (residual learning)
        if target is None:
            target = attn_contextualized

        result = self.ffn.fit(
            attn_contextualized,
            target,
            use_low_rank=use_low_rank,
        )

        self._ffn_fitted = result.get("success", False)

        return result

    @property
    def is_ffn_fitted(self) -> bool:
        return self._ffn_fitted


class AttentionPILModel(nn.Module):
    """
    Full Attention-PIL Language Model.

    Architecture:
        - Token Embedding (trainable via backprop)
        - Positional Encoding
        - N x AttentionPILBlock
        - Output Head (Linear projection to vocab)

    Training Strategy:
        1. Phase 1: Fit all PIL FFN layers using .fit_all_ffn()
        2. Phase 2: (Optional) Fine-tune attention via backprop
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        max_seq_len: int = 512,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        freeze_attention: bool = False,
        ffn_type: Literal["bipil", "swarm"] = "bipil",
        reg_lambda: float = 1e-5,
    ):
        """
        Initialize Attention-PIL Model.

        Args:
            vocab_size: Vocabulary size
            dim: Model dimension
            n_layers: Number of transformer blocks
            n_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            expansion_factor: FFN expansion factor
            dropout: Dropout rate
            freeze_attention: Freeze all attention weights
            ffn_type: Type of PIL FFN
            reg_lambda: Ridge regression parameter
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Token Embedding (CAN be trained via backprop)
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Positional Encoding (learned)
        self.pos_encoding = nn.Embedding(max_seq_len, dim)

        # Transformer Blocks
        self.blocks = nn.ModuleList(
            [
                AttentionPILBlock(
                    dim=dim,
                    n_heads=n_heads,
                    expansion_factor=expansion_factor,
                    dropout=dropout,
                    freeze_attention=freeze_attention,
                    ffn_type=ffn_type,
                    reg_lambda=reg_lambda,
                )
                for _ in range(n_layers)
            ]
        )

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(dim)

        # Output Head (projects to vocab)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights with embedding
        self.output_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)

        # Initialize
        self._init_weights()

        self.monitor = NumericalMonitor()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_encoding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (B, S)
            mask: Attention mask

        Returns:
            Logits (B, S, vocab_size)
        """
        B, S = input_ids.shape

        # Embeddings
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.pos_encoding(positions)
        x = self.dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final norm
        x = self.final_norm(x)

        # Project to vocab
        logits = self.output_head(x)

        return logits

    @torch.no_grad()
    def fit_all_ffn(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_low_rank: bool = False,
        layer_by_layer: bool = True,
    ) -> dict:
        """
        Fit all PIL FFN layers in the model.

        This is the main training function for PIL - NO BACKPROP for FFNs.

        Args:
            input_ids: Input token IDs (B, S)
            target_ids: Target token IDs (for next-token prediction). If None, uses input_ids shifted.
            mask: Attention mask
            use_low_rank: Use low-rank approximation
            layer_by_layer: Fit layers sequentially (recommended)

        Returns:
            Training statistics for all layers
        """
        B, S = input_ids.shape

        # Get embeddings
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.pos_encoding(positions)

        # Prepare target embeddings if provided
        if target_ids is not None:
            target_emb = self.token_embedding(target_ids) + self.pos_encoding(positions)
        else:
            target_emb = None

        results = []

        if layer_by_layer:
            # Sequential fitting: each layer uses output of previous layer as input
            current_x = x

            for i, block in enumerate(self.blocks):
                # Compute attention output (this runs attention forward)
                attn_out, _ = block.attention(current_x, mask)
                attn_x = block.attn_norm(current_x + attn_out)

                # Fit FFN: target is the expected output
                # For residual learning, target = attn_x
                result = block.ffn.fit(
                    attn_x,
                    target=attn_x if target_emb is None else target_emb,
                    use_low_rank=use_low_rank,
                )
                result["layer"] = i
                results.append(result)

                # Update current_x for next layer
                current_x = block.ffn(attn_x)

                logger.info(
                    f"layer_{i}_fitted",
                    success=result.get("success"),
                    mse=result.get("mse"),
                )

        else:
            # Parallel fitting: all layers see same input (faster but less accurate)
            for i, block in enumerate(self.blocks):
                result = block.fit_ffn(x, target_emb, mask, use_low_rank)
                result["layer"] = i
                results.append(result)

        # Summary
        successful = sum(1 for r in results if r.get("success", False))
        avg_mse = sum(r.get("mse", 0) for r in results) / len(results)

        return {
            "layers_fitted": successful,
            "total_layers": len(results),
            "avg_mse": avg_mse,
            "layer_results": results,
        }

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters (for attention/embedding only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get total parameters including solved PIL weights."""
        return sum(p.numel() for p in self.parameters())


class PILTrainer:
    """
    Training harness for Attention-PIL Model.

    Implements the correct training sequence:
        1. Buffer batches of data
        2. Fit PIL FFN layers using pseudoinverse
        3. (Optional) Update attention/embeddings via backprop
    """

    def __init__(
        self,
        model: AttentionPILModel,
        learning_rate: float = 1e-4,
        use_attention_backprop: bool = True,
    ):
        """
        Initialize PIL Trainer.

        Args:
            model: AttentionPILModel to train
            learning_rate: LR for attention/embedding gradients
            use_attention_backprop: Whether to train attention via backprop
        """
        self.model = model
        self.use_attention_backprop = use_attention_backprop

        # Optimizer for attention/embeddings ONLY
        if use_attention_backprop:
            attention_params = []
            for block in model.blocks:
                if not block.attention.freeze:
                    attention_params.extend(block.attention.parameters())

            # Also include embeddings
            attention_params.extend(model.token_embedding.parameters())
            attention_params.extend(model.pos_encoding.parameters())

            self.optimizer = torch.optim.AdamW(attention_params, lr=learning_rate)
        else:
            self.optimizer = None

        self.epoch = 0
        self.step = 0
        self.history = {
            "pil_mse": [],
            "attn_loss": [],
        }

    def fit_epoch(
        self,
        dataloader,
        use_low_rank: bool = False,
    ) -> dict:
        """
        Run one epoch of PIL fitting.

        For each batch:
            1. Fit PIL FFN layers (no backprop)
            2. (Optional) Update attention via backprop

        Args:
            dataloader: Yields (input_ids, target_ids, mask) tuples
            use_low_rank: Use low-rank approximation

        Returns:
            Epoch statistics
        """
        self.model.train()

        epoch_pil_mse = []
        epoch_attn_loss = []

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch[0]
            target_ids = batch[1] if len(batch) > 1 else None
            mask = batch[2] if len(batch) > 2 else None

            # Step 1: Fit PIL FFN layers (NO BACKPROP)
            pil_result = self.model.fit_all_ffn(
                input_ids,
                target_ids,
                mask,
                use_low_rank=use_low_rank,
            )
            epoch_pil_mse.append(pil_result["avg_mse"])

            # Step 2: (Optional) Update attention via backprop
            if self.use_attention_backprop and self.optimizer is not None:
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(input_ids, mask)

                # Loss for language modeling
                if target_ids is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, self.model.vocab_size),
                        target_ids.view(-1),
                        ignore_index=-100,
                    )
                else:
                    # Self-supervised: predict next token
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, self.model.vocab_size),
                        shift_labels.view(-1),
                    )

                # Backprop for attention ONLY
                loss.backward()
                self.optimizer.step()

                epoch_attn_loss.append(loss.item())

            self.step += 1

            if batch_idx % 10 == 0:
                logger.info(
                    f"batch_{batch_idx}",
                    pil_mse=pil_result["avg_mse"],
                    attn_loss=epoch_attn_loss[-1] if epoch_attn_loss else None,
                )

        self.epoch += 1

        avg_pil_mse = sum(epoch_pil_mse) / len(epoch_pil_mse) if epoch_pil_mse else 0
        avg_attn_loss = (
            sum(epoch_attn_loss) / len(epoch_attn_loss) if epoch_attn_loss else 0
        )

        self.history["pil_mse"].append(avg_pil_mse)
        self.history["attn_loss"].append(avg_attn_loss)

        return {
            "epoch": self.epoch,
            "avg_pil_mse": avg_pil_mse,
            "avg_attn_loss": avg_attn_loss,
            "total_steps": self.step,
        }

    def get_history(self) -> dict:
        """Get training history."""
        return self.history
