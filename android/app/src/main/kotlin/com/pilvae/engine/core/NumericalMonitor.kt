package com.pilvae.engine.core

import android.util.Log

/**
 * Monitor numerical stability during PIL training.
 *
 * Tracks condition numbers, NaN occurrences, and other stability metrics.
 * Kotlin port of NumericalMonitor class from pil_utils.py
 */
class NumericalMonitor(
    private val warnThreshold: Float = 1e8f,
    private val failThreshold: Float = 1e12f
) {
    private val TAG = "NumericalMonitor"

    data class History(
        val conditionNumbers: MutableList<Float> = mutableListOf(),
        var nanCount: Int = 0,
        var infCount: Int = 0,
        val warnings: MutableList<String> = mutableListOf()
    )

    private var history = History()

    /**
     * Check a matrix for numerical issues.
     *
     * @param matrix Matrix to check
     * @param name Name for logging
     * @return True if matrix is stable, False otherwise
     */
    fun checkMatrix(matrix: Array<FloatArray>, name: String = "matrix"): Boolean {
        // Check for NaN
        for (row in matrix) {
            for (value in row) {
                if (value.isNaN()) {
                    history.nanCount++
                    Log.e(TAG, "NaN detected in $name")
                    return false
                }
                if (value.isInfinite()) {
                    history.infCount++
                    Log.e(TAG, "Inf detected in $name")
                    return false
                }
            }
        }

        // Check condition number for square matrices
        if (matrix.size == matrix[0].size) {
            val cond = PILUtils.conditionNumber(matrix)
            history.conditionNumbers.add(cond)

            when {
                cond > failThreshold -> {
                    Log.e(TAG, "$name condition critical: κ = $cond")
                    return false
                }
                cond > warnThreshold -> {
                    history.warnings.add("$name: κ=${String.format("%.2e", cond)}")
                    Log.w(TAG, "$name condition warning: κ = $cond")
                }
            }
        }

        return true
    }

    /**
     * Check a vector for numerical issues.
     */
    fun checkVector(vector: FloatArray, name: String = "vector"): Boolean {
        for (value in vector) {
            if (value.isNaN()) {
                history.nanCount++
                Log.e(TAG, "NaN detected in $name")
                return false
            }
            if (value.isInfinite()) {
                history.infCount++
                Log.e(TAG, "Inf detected in $name")
                return false
            }
        }
        return true
    }

    /**
     * Get summary of numerical stability.
     */
    fun getSummary(): Map<String, Any> {
        val condNums = history.conditionNumbers
        return mapOf(
            "nan_count" to history.nanCount,
            "inf_count" to history.infCount,
            "warning_count" to history.warnings.size,
            "max_condition_number" to (condNums.maxOrNull() ?: 0f),
            "mean_condition_number" to if (condNums.isNotEmpty()) condNums.average() else 0.0
        )
    }

    /**
     * Reset monitoring history.
     */
    fun reset() {
        history = History()
    }

    /**
     * Get detailed report as string.
     */
    fun getReport(): String {
        val summary = getSummary()
        return buildString {
            appendLine("=== Numerical Stability Report ===")
            appendLine("NaN Count: ${summary["nan_count"]}")
            appendLine("Inf Count: ${summary["inf_count"]}")
            appendLine("Warnings: ${summary["warning_count"]}")
            appendLine("Max Condition Number: ${String.format("%.2e", summary["max_condition_number"])}")
            appendLine("Mean Condition Number: ${String.format("%.2e", summary["mean_condition_number"])}")

            if (history.warnings.isNotEmpty()) {
                appendLine("\nWarnings:")
                history.warnings.forEach { appendLine("  - $it") }
            }
        }
    }
}
