package com.pilvae.engine.core

import android.util.Log
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * PIL-VAE: Pseudoinverse Learning Variational Autoencoder.
 *
 * Hybrid architecture combining:
 * - Gradient-free PIL for Feed-Forward layers
 * - Standard encoding for embeddings
 *
 * Kotlin port of pil_vae.py - maintains exact algorithmic structure.
 */
class PILVAE(
    private val inputDim: Int,
    private val latentDim: Int,
    private val hiddenDim: Int = 128,
    private val regLambda: Float = 1e-5f,
    private val seed: Long = 42L
) {
    private val TAG = "PILVAE"

    // Encoder: Maps input to latent space (mean and logvar)
    private val encoderLayer: BiPILLayer

    // Decoder: Maps latent space back to input
    private val decoderLayer: BiPILLayer

    // Numerical monitoring
    private val monitor = NumericalMonitor()

    init {
        Log.d(TAG, "Initializing PIL-VAE: input=$inputDim, latent=$latentDim, hidden=$hiddenDim")

        // Encoder: input_dim -> latent_dim * 2 (mean + logvar)
        encoderLayer = BiPILLayer(
            inputDim = inputDim,
            hiddenDim = hiddenDim,
            outputDim = latentDim * 2,
            regLambda = regLambda,
            seed = seed
        )

        // Decoder: latent_dim -> input_dim
        decoderLayer = BiPILLayer(
            inputDim = latentDim,
            hiddenDim = hiddenDim,
            outputDim = inputDim,
            regLambda = regLambda,
            seed = seed + 100
        )
    }

    /**
     * Encode input to latent distribution parameters.
     *
     * @param x Input tensor (batch_size, input_dim)
     * @return Pair of (mean, logvar) each (batch_size, latent_dim)
     */
    fun encode(x: Array<FloatArray>): Pair<Array<FloatArray>, Array<FloatArray>> {
        val output = encoderLayer.forward(x)

        // Split into mean and logvar
        val mean = Array(x.size) { i ->
            FloatArray(latentDim) { j -> output[i][j] }
        }
        val logvar = Array(x.size) { i ->
            FloatArray(latentDim) { j -> output[i][j + latentDim] }
        }

        return Pair(mean, logvar)
    }

    /**
     * Reparameterization trick: z = mean + std * epsilon
     */
    fun reparameterize(mean: Array<FloatArray>, logvar: Array<FloatArray>): Array<FloatArray> {
        val random = java.util.Random()

        return Array(mean.size) { i ->
            FloatArray(latentDim) { j ->
                val std = exp(0.5f * logvar[i][j])
                val eps = random.nextGaussian().toFloat()
                mean[i][j] + std * eps
            }
        }
    }

    /**
     * Decode latent vectors to reconstruction.
     *
     * @param z Latent tensor (batch_size, latent_dim)
     * @return Reconstruction (batch_size, input_dim)
     */
    fun decode(z: Array<FloatArray>): Array<FloatArray> {
        return decoderLayer.forward(z)
    }

    /**
     * Full forward pass: encode -> reparameterize -> decode
     *
     * @param x Input tensor (batch_size, input_dim)
     * @return Triple of (reconstruction, mean, logvar)
     */
    fun forward(x: Array<FloatArray>): Triple<Array<FloatArray>, Array<FloatArray>, Array<FloatArray>> {
        val (mean, logvar) = encode(x)
        val z = reparameterize(mean, logvar)
        val reconstruction = decode(z)
        return Triple(reconstruction, mean, logvar)
    }

    /**
     * Fit the VAE using PIL (no backpropagation).
     *
     * Two-stage fitting:
     * 1. Fit encoder to produce good latent representations
     * 2. Fit decoder to reconstruct from latents
     *
     * @param x Training data (batch_size, input_dim)
     * @return True if fitting succeeded
     */
    fun fit(x: Array<FloatArray>): Boolean {
        Log.d(TAG, "Fitting PIL-VAE on batch of size ${x.size}")

        // Stage 1: Fit encoder
        // Target: We want latent codes that can reconstruct x
        // For initial fitting, we use a simple target (normalized x)
        val encoderTarget = normalizeRows(x).let { normalized ->
            // Expand to latent_dim * 2 (mean + logvar)
            Array(x.size) { i ->
                FloatArray(latentDim * 2) { j ->
                    if (j < latentDim) {
                        normalized[i][j % normalized[i].size]
                    } else {
                        -1f // Initial logvar (small variance)
                    }
                }
            }
        }

        if (!encoderLayer.fit(x, encoderTarget)) {
            Log.e(TAG, "Encoder fitting failed")
            return false
        }

        // Stage 2: Encode to get latents, then fit decoder
        val (mean, _) = encode(x)

        if (!decoderLayer.fit(mean, x)) {
            Log.e(TAG, "Decoder fitting failed")
            return false
        }

        Log.d(TAG, "PIL-VAE fitting complete")
        return true
    }

    /**
     * Compute VAE loss: Reconstruction + KL divergence.
     */
    fun computeLoss(x: Array<FloatArray>): Map<String, Float> {
        val (recon, mean, logvar) = forward(x)

        // Reconstruction loss (MSE)
        var reconLoss = 0f
        var count = 0
        for (i in x.indices) {
            for (j in x[i].indices) {
                val diff = recon[i][j] - x[i][j]
                reconLoss += diff * diff
                count++
            }
        }
        reconLoss = if (count > 0) reconLoss / count else 0f

        // KL divergence: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        var klLoss = 0f
        for (i in mean.indices) {
            for (j in 0 until latentDim) {
                klLoss += -0.5f * (1f + logvar[i][j] - mean[i][j] * mean[i][j] - exp(logvar[i][j]))
            }
        }
        klLoss /= mean.size

        return mapOf(
            "reconstruction_loss" to reconLoss,
            "kl_loss" to klLoss,
            "total_loss" to reconLoss + klLoss
        )
    }

    /**
     * Generate new samples by decoding random latent vectors.
     */
    fun generate(numSamples: Int): Array<FloatArray> {
        val random = java.util.Random()
        val z = Array(numSamples) {
            FloatArray(latentDim) { random.nextGaussian().toFloat() }
        }
        return decode(z)
    }

    /**
     * Get latent representation of input.
     */
    fun getLatent(x: Array<FloatArray>): Array<FloatArray> {
        val (mean, _) = encode(x)
        return mean
    }

    /**
     * Get numerical stability report.
     */
    fun getStabilityReport(): String {
        return buildString {
            appendLine("=== PIL-VAE Stability Report ===")
            appendLine("\nEncoder:")
            appendLine(encoderLayer.getStabilityReport())
            appendLine("\nDecoder:")
            appendLine(decoderLayer.getStabilityReport())
        }
    }

    // ============================================================================
    // Private Helper Methods
    // ============================================================================

    /**
     * Normalize rows to unit norm.
     */
    private fun normalizeRows(matrix: Array<FloatArray>): Array<FloatArray> {
        return Array(matrix.size) { i ->
            val norm = sqrt(matrix[i].map { it * it }.sum())
            if (norm > 1e-10f) {
                FloatArray(matrix[i].size) { j -> matrix[i][j] / norm }
            } else {
                matrix[i].copyOf()
            }
        }
    }
}
