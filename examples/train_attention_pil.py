"""
Example Training Script for Attention-PIL Hybrid Architecture.

This demonstrates the correct training sequence for PIL:
    1. NO BACKPROP for FFN weights - they are solved via pseudoinverse
    2. Attention/Embeddings CAN use gradients (optional)

Usage:
    python examples/train_attention_pil.py

Reference: Project Emergent-1 PRD
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import structlog

# Import PIL components
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.attention_pil import AttentionPILModel, PILTrainer
from app.core.pil_utils import NumericalMonitor

logger = structlog.get_logger(__name__)


class SimpleTextDataset(Dataset):
    """Simple dataset for demonstration."""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples

        # Generate random sequences for demonstration
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return (input, target) - target is shifted by 1 for next-token prediction
        input_ids = self.data[idx]
        target_ids = torch.roll(input_ids, -1)
        target_ids[-1] = 0  # Pad last token
        return input_ids, target_ids


def train_pil_model():
    """
    Main training function demonstrating PIL training sequence.

    Key Points:
        - FFN weights are SOLVED via .fit() method (NO loss.backward())
        - Attention weights MAY use gradients (configurable)
        - Training is "one-shot" per batch for FFN layers
    """

    # Configuration
    config = {
        "vocab_size": 5000,
        "dim": 128,
        "n_layers": 3,
        "n_heads": 4,
        "expansion_factor": 4,
        "max_seq_len": 64,
        "reg_lambda": 1e-5,
        "freeze_attention": False,  # Allow attention gradients
        "learning_rate": 1e-4,
        "batch_size": 16,
        "n_epochs": 5,
        "n_train_samples": 1000,
    }

    logger.info("config", **config)

    # Create model
    model = AttentionPILModel(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        expansion_factor=config["expansion_factor"],
        freeze_attention=config["freeze_attention"],
        reg_lambda=config["reg_lambda"],
    )

    logger.info(
        "model_created",
        total_params=model.get_total_params(),
        trainable_params=model.get_trainable_params(),
    )

    # Create dataset and dataloader
    dataset = SimpleTextDataset(
        vocab_size=config["vocab_size"],
        seq_len=config["max_seq_len"],
        n_samples=config["n_train_samples"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    # Create trainer
    trainer = PILTrainer(
        model=model,
        learning_rate=config["learning_rate"],
        use_attention_backprop=not config["freeze_attention"],
    )

    # Training loop
    logger.info("training_started")

    total_start = time.time()

    for epoch in range(config["n_epochs"]):
        epoch_start = time.time()

        stats = trainer.fit_epoch(dataloader, use_low_rank=False)

        epoch_time = time.time() - epoch_start
        tokens_per_sec = config["n_train_samples"] * config["max_seq_len"] / epoch_time

        logger.info(
            f"epoch_{epoch + 1}",
            pil_mse=f"{stats['avg_pil_mse']:.6f}",
            attn_loss=f"{stats['avg_attn_loss']:.4f}"
            if stats["avg_attn_loss"]
            else "N/A",
            time_sec=f"{epoch_time:.2f}",
            tokens_per_sec=f"{tokens_per_sec:.0f}",
        )

    total_time = time.time() - total_start

    logger.info(
        "training_completed",
        total_time_sec=f"{total_time:.2f}",
        final_pil_mse=trainer.history["pil_mse"][-1],
    )

    return model, trainer


def demonstrate_fit_sequence():
    """
    Demonstrate the correct PIL .fit() sequence.

    This is the CORRECT way to train PIL layers:
        1. Collect batch of (input, target) pairs
        2. Call .fit() method (NOT optimizer.step())
        3. Weights are solved via pseudoinverse in ONE SHOT
    """

    logger.info("=== Demonstrating PIL Fit Sequence ===")

    # Create a simple BiPIL layer
    from app.core.bipil_layer import BiPILLayer

    layer = BiPILLayer(
        dim=64,
        expansion_factor=4,
        reg_lambda=1e-5,
        seed=42,
    )

    # Create training data
    x = torch.randn(100, 64)  # 100 samples, 64-dim
    target = torch.randn(100, 64)  # Target values

    logger.info("before_fit", is_fitted=layer.is_fitted)

    # ============================================
    # CORRECT: Use .fit() method (NO BACKPROP)
    # ============================================
    result = layer.fit(x, target)

    logger.info(
        "after_fit",
        is_fitted=layer.is_fitted,
        method=result["method"],
        mse=f"{result['mse']:.6f}",
        condition_number=f"{result['condition_number']:.2e}",
    )

    # ============================================
    # INCORRECT: DO NOT DO THIS FOR PIL LAYERS
    # ============================================
    # optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
    # loss = criterion(layer(x), target)
    # loss.backward()  # WRONG! PIL doesn't use backprop
    # optimizer.step()  # WRONG!

    # Forward pass after fitting
    y = layer(x.unsqueeze(0))  # Add batch dim
    logger.info("forward_pass", output_shape=y.shape)


def demonstrate_full_model_training():
    """
    Demonstrate full model training sequence.

    Key difference from standard PyTorch training:
        - PIL FFN layers use .fit_all_ffn() method
        - Only attention/embedding gradients use optimizer
    """

    logger.info("=== Demonstrating Full Model Training ===")

    # Create mini model
    model = AttentionPILModel(
        vocab_size=1000,
        dim=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
    )

    # Sample batch
    input_ids = torch.randint(0, 1000, (8, 32))

    # ============================================
    # STEP 1: Fit PIL FFN layers (ONE-SHOT, NO BACKPROP)
    # ============================================
    fit_result = model.fit_all_ffn(input_ids)

    logger.info(
        "pil_ffn_fitted",
        layers_fitted=fit_result["layers_fitted"],
        avg_mse=f"{fit_result['avg_mse']:.6f}",
    )

    # ============================================
    # STEP 2: (Optional) Update attention via backprop
    # ============================================
    # Only attention and embedding parameters use gradients
    attention_params = []
    for block in model.blocks:
        attention_params.extend(block.attention.parameters())
    attention_params.extend(model.token_embedding.parameters())

    optimizer = torch.optim.AdamW(attention_params, lr=1e-4)

    # Forward pass
    logits = model(input_ids)

    # Compute loss (next-token prediction)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, 1000),
        shift_labels.view(-1),
    )

    # Backprop for attention ONLY
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.info("attention_updated", loss=f"{loss.item():.4f}")


if __name__ == "__main__":
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    # Run demonstrations
    print("\n" + "=" * 60)
    print("Project Emergent-1: Attention-PIL Training Demo")
    print("=" * 60 + "\n")

    # Demo 1: Simple fit sequence
    demonstrate_fit_sequence()

    print("\n" + "-" * 60 + "\n")

    # Demo 2: Full model training
    demonstrate_full_model_training()

    print("\n" + "-" * 60 + "\n")

    # Demo 3: Full training loop
    print("Running full training loop...")
    model, trainer = train_pil_model()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
