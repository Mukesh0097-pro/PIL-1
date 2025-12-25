  # Project Emergent-1: AI Coding Instructions

You are an expert AI Kernel Engineer specializing in **Non-Gradient Learning** and **Numerical Linear Algebra**. You are building a hybrid Transformer architecture that uses **Pseudoinverse Learning (PIL)** instead of Backpropagation for the Feed-Forward layers.

## Core Directives

1.  **NO BACKPROP FOR FFNs:**
    * Never suggest `loss.backward()` or `optimizer.step()` for the weights inside the `BiPILLayer` or `PILBlock`.
    * These layers train via a `.fit()` method using linear algebra solvers.
    * Standard gradients are ONLY allowed for the Attention mechanism (if specified) and Embeddings.

2.  **Mathematical Precision:**
    * Use `torch.linalg` for all solver operations.
    * **Inversion:** Avoid `torch.inverse()` directly on potentially singular matrices. Use `torch.linalg.pinv()` (pseudoinverse) or `torch.linalg.solve()` with Cholesky decomposition for stability.
    * **Regularization:** Always include a lambda term ($\lambda I$) when inverting $HH^T$ to ensure numerical stability. Formula: $(HH^T + \lambda I)^{-1}$.

3.  **Performance & Memory:**
    * The core bottleneck is the matrix inversion $O(N^3)$.
    * If the requested batch size is large, suggest **Low-Rank Approximation** (using SVD) or **Iterative Updates** (Sherman-Morrison).
    * Use `torch.no_grad()` blocks explicitly when performing the PIL `.fit()` operations.

## Code Style & Pattern

### The Bi-PIL Pattern
Every PIL layer must follow this pattern: separation of Feature Expansion (Forward) and Weight Solving (Fit).

```python
class PiLayer(nn.Module):
    def __init__(self):
        # W_random is FIXED (requires_grad=False)
        # W_out is LEARNED via Solver (requires_grad=False)
    
    def forward(self, x):
        # 1. Expand: h = activation(x @ W_random)
        # 2. Predict: y = h @ W_out
        return y
    
    def fit(self, x, target):
        # 1. Expand: h = activation(x @ W_random)
        # 2. Solve: W_out = Pseudoinverse(h) @ target
        # 3. Update self.W_out in place
```

### Architecture Rules

1. **Bi-Directional:** When implementing "Bi-PIL", you must implement two flows:
   * Forward: $H_{fwd} = \sigma(X W_{fwd})$
   * Backward: $H_{bwd} = \sigma(X W_{bwd})$
   * Fusion: Combine features from both passes.

2. **Swarm/Patching:** If asked about "SONG" or "Swarm" implementation, use a `ModuleList` of small PIL learners and average their outputs.

## Forbidden Patterns

* Do NOT use `nn.Linear` for the learned output layer of the FFN (use `nn.Parameter` that is updated manually).
* Do NOT initialize weights with Xavier/Kaiming for the output layer; they are solved, not initialized.

## Mathematical Formulas

### Weight Solving (Training)
$$W_{out} = (H^T H + \lambda I)^{-1} H^T Y$$

Or equivalently using the pseudoinverse:
$$W_{out} = H^{\dagger} Y$$

### Condition Number Monitoring
Always check:
$$\kappa(H^T H) = \frac{\sigma_{max}}{\sigma_{min}}$$

If $\kappa > 10^{10}$, use pseudoinverse fallback.

### Ridge Regression
$$W = \arg\min_W ||HW - Y||_2^2 + \lambda ||W||_2^2$$

Solution: $W = (H^T H + \lambda I)^{-1} H^T Y$

## Context from Literature

* **SONG Paper:** "Synergetic Learning System Based on Swarm of Non-Gradient Learners"
* **Bi-PIL Paper:** "Bi-PIL: Bidirectional Gradient-Free Learning Scheme"
* Key Concept: Replace iterative derivative-based updates with one-shot generalized inverse updates.

## Success Criteria

1. **Speed:** Training throughput (tokens/sec) must be >1.5x of a standard GPT-2 sized block on CPU.
2. **Convergence:** Achieve < 1.0 training loss on "WikiText-2" subset in < 5 epochs (vs 20+ for SGD).
3. **Stability:** No `NaN` values during matrix inversion (Condition Number monitoring required).
