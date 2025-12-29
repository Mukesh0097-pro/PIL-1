package com.pilvae.engine.core

import android.util.Log
import kotlin.math.tanh

/**
 * Bi-PIL Layer: Bidirectional Pseudoinverse Learning.
 *
 * Implements gradient-free learning using matrix inversion.
 * W_random is FIXED (requires_grad=False), W_out is SOLVED via linear algebra.
 *
 * Kotlin port of bipil_layer.py - mirrors the exact algorithm structure.
 *
 * Key PIL Principle: NO BACKPROP FOR FFNs
 * - These layers train via .fit() method using linear algebra solvers
 * - Standard gradients are NOT used
 */
class BiPILLayer(
    private val inputDim: Int,
    private val hiddenDim: Int,
    private val outputDim: Int,
    private val regLambda: Float = 1e-5f,
    private val seed: Long = 42L
) {
    private val TAG = "BiPILLayer"

    // Fixed random expansion weights (requires_grad = false equivalent)
    // These are NEVER trained - they provide random feature expansion
    private val wRandomFwd: Array<FloatArray>
    private val wRandomBwd: Array<FloatArray>

    // Learned output weights (SOLVED via linear algebra, not trained with gradients)
    private var wOut: Array<FloatArray>

    // For Sherman-Morrison incremental updates
    private var hInv: Array<FloatArray>? = null

    // Numerical stability monitor
    private val monitor = NumericalMonitor()

    init {
        Log.d(TAG, "Initializing BiPIL: input=$inputDim, hidden=$hiddenDim, output=$outputDim")

        // Initialize fixed random weights with orthogonal initialization
        wRandomFwd = PILUtils.orthogonalInit(inputDim, hiddenDim, seed)
        wRandomBwd = PILUtils.orthogonalInit(inputDim, hiddenDim, seed + 1)

        // Initialize output weights to zeros (will be solved, not trained)
        wOut = PILUtils.zeros(hiddenDim * 2, outputDim)
    }

    /**
     * Forward pass: Expand features bidirectionally and predict.
     *
     * h_fwd = tanh(X @ W_fwd)
     * h_bwd = tanh(X @ W_bwd)
     * h = concat(h_fwd, h_bwd)
     * y = h @ W_out
     *
     * @param x Input tensor (batch_size, input_dim)
     * @return Output tensor (batch_size, output_dim)
     */
    fun forward(x: Array<FloatArray>): Array<FloatArray> {
        val batchSize = x.size

        // Forward expansion: h_fwd = tanh(x @ W_random_fwd)
        val hFwd = activate(PILUtils.matmul(x, wRandomFwd))

        // Backward expansion: h_bwd = tanh(x @ W_random_bwd)
        val hBwd = activate(PILUtils.matmul(x, wRandomBwd))

        // Concatenate: (batch, hidden * 2)
        val hConcat = Array(batchSize) { i ->
            FloatArray(hiddenDim * 2) { j ->
                if (j < hiddenDim) hFwd[i][j] else hBwd[i][j - hiddenDim]
            }
        }

        // Predict: y = h @ W_out
        return PILUtils.matmul(hConcat, wOut)
    }

    /**
     * Fit weights using ridge regression (ONE-SHOT, NO BACKPROP).
     *
     * Core PIL Training Formula:
     * W_out = (H^T H + λI)^{-1} H^T Y
     *
     * @param x Input tensor (batch_size, input_dim)
     * @param target Target tensor (batch_size, output_dim)
     * @return True if fitting succeeded
     */
    fun fit(x: Array<FloatArray>, target: Array<FloatArray>): Boolean {
        val batchSize = x.size
        Log.d(TAG, "Fitting on batch of size $batchSize")

        // Expand features (same as forward pass)
        val hFwd = activate(PILUtils.matmul(x, wRandomFwd))
        val hBwd = activate(PILUtils.matmul(x, wRandomBwd))

        // Concatenate
        val hConcat = Array(batchSize) { i ->
            FloatArray(hiddenDim * 2) { j ->
                if (j < hiddenDim) hFwd[i][j] else hBwd[i][j - hiddenDim]
            }
        }

        // Check numerical stability
        if (!monitor.checkMatrix(hConcat, "hidden_activations")) {
            Log.e(TAG, "Numerical instability in hidden activations")
            return false
        }

        // Solve for W_out using ridge regression
        // W = (H^T H + λI)^{-1} H^T Y
        val newWOut = PILUtils.ridgeSolve(hConcat, target, regLambda)

        return if (newWOut != null) {
            if (!monitor.checkMatrix(newWOut, "solved_weights")) {
                Log.e(TAG, "Numerical instability in solved weights")
                false
            } else {
                wOut = newWOut
                Log.d(TAG, "Weights successfully solved")
                true
            }
        } else {
            Log.e(TAG, "Ridge solve failed")
            false
        }
    }

    /**
     * Incremental update using Sherman-Morrison formula.
     * For online learning, one sample at a time.
     *
     * @param x Single input (input_dim,)
     * @param target Single target (output_dim,)
     * @return True if update succeeded
     */
    fun incrementalUpdate(x: FloatArray, target: FloatArray): Boolean {
        // Expand single sample
        val hFwd = activateVector(matVecLocal(wRandomFwd, x))
        val hBwd = activateVector(matVecLocal(wRandomBwd, x))

        // Concatenate
        val hConcat = FloatArray(hiddenDim * 2) { j ->
            if (j < hiddenDim) hFwd[j] else hBwd[j - hiddenDim]
        }

        // Initialize H_inv if needed
        if (hInv == null) {
            hInv = PILUtils.eye(hiddenDim * 2).map { row ->
                row.map { it / regLambda }.toFloatArray()
            }.toTypedArray()
        }

        // Sherman-Morrison update
        val result = PILUtils.shermanMorrisonUpdate(wOut, hInv!!, hConcat, target, regLambda)

        wOut = result.W
        hInv = result.HInv

        return true
    }

    /**
     * Get current weights for inspection.
     */
    fun getWeights(): Map<String, Array<FloatArray>> {
        return mapOf(
            "w_random_fwd" to wRandomFwd,
            "w_random_bwd" to wRandomBwd,
            "w_out" to wOut
        )
    }

    /**
     * Compute loss (MSE) for evaluation.
     */
    fun computeLoss(x: Array<FloatArray>, target: Array<FloatArray>): Float {
        val predictions = forward(x)
        var mse = 0f
        var count = 0

        for (i in predictions.indices) {
            for (j in predictions[i].indices) {
                val diff = predictions[i][j] - target[i][j]
                mse += diff * diff
                count++
            }
        }

        return if (count > 0) mse / count else 0f
    }

    /**
     * Get numerical stability report.
     */
    fun getStabilityReport(): String = monitor.getReport()

    /**
     * Reset numerical monitor.
     */
    fun resetMonitor() = monitor.reset()

    // ============================================================================
    // Private Helper Methods
    // ============================================================================

    /**
     * Tanh activation function for matrices.
     */
    private fun activate(matrix: Array<FloatArray>): Array<FloatArray> {
        return Array(matrix.size) { i ->
            FloatArray(matrix[i].size) { j ->
                tanh(matrix[i][j].toDouble()).toFloat()
            }
        }
    }

    /**
     * Tanh activation function for vectors.
     */
    private fun activateVector(vec: FloatArray): FloatArray {
        return FloatArray(vec.size) { i ->
            tanh(vec[i].toDouble()).toFloat()
        }
    }

    /**
     * Matrix^T @ vector (for single sample)
     */
    private fun matVecLocal(matrix: Array<FloatArray>, vec: FloatArray): FloatArray {
        val rows = matrix.size
        val cols = matrix[0].size
        return FloatArray(cols) { j ->
            var sum = 0f
            for (i in 0 until rows) {
                sum += matrix[i][j] * vec[i]
            }
            sum
        }
    }
}
