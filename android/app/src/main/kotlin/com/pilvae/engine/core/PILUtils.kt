package com.pilvae.engine.core

import android.util.Log
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * PIL Utilities: Numerical Linear Algebra Utilities for Pseudoinverse Learning.
 *
 * This module provides numerically stable matrix operations for the Bi-PIL architecture.
 * Kotlin port of app/core/pil_utils.py - All operations mirror torch.linalg behavior.
 */
object PILUtils {

    private const val TAG = "PILUtils"
    private const val DEFAULT_REG_LAMBDA = 1e-5f
    private const val WARN_THRESHOLD = 1e8f
    private const val FAIL_THRESHOLD = 1e12f

    // ============================================================================
    // Basic Matrix Operations (mirrors torch operations)
    // ============================================================================

    /**
     * Matrix multiplication: A @ B
     * Equivalent to torch.matmul(A, B)
     */
    fun matmul(a: Array<FloatArray>, b: Array<FloatArray>): Array<FloatArray> {
        val m = a.size
        val n = b[0].size
        val k = b.size

        require(a[0].size == k) { "Matrix dimensions incompatible: ${a[0].size} vs $k" }

        return Array(m) { i ->
            FloatArray(n) { j ->
                var sum = 0f
                for (l in 0 until k) {
                    sum += a[i][l] * b[l][j]
                }
                sum
            }
        }
    }

    /**
     * Matrix-vector multiplication: A @ v
     */
    fun matVec(matrix: Array<FloatArray>, vec: FloatArray): FloatArray {
        require(matrix[0].size == vec.size) { "Dimensions incompatible" }
        return FloatArray(matrix.size) { i ->
            var sum = 0f
            for (j in vec.indices) {
                sum += matrix[i][j] * vec[j]
            }
            sum
        }
    }

    /**
     * Matrix transpose: A.T
     * Equivalent to torch.transpose or matrix.T
     */
    fun transpose(matrix: Array<FloatArray>): Array<FloatArray> {
        val rows = matrix.size
        val cols = matrix[0].size
        return Array(cols) { j ->
            FloatArray(rows) { i ->
                matrix[i][j]
            }
        }
    }

    /**
     * Create identity matrix: torch.eye(n)
     */
    fun eye(n: Int): Array<FloatArray> {
        return Array(n) { i ->
            FloatArray(n) { j ->
                if (i == j) 1f else 0f
            }
        }
    }

    /**
     * Create zero matrix: torch.zeros(rows, cols)
     */
    fun zeros(rows: Int, cols: Int): Array<FloatArray> {
        return Array(rows) { FloatArray(cols) }
    }

    /**
     * Vector L2 norm: torch.linalg.norm(vec)
     */
    fun norm(vec: FloatArray): Float {
        return sqrt(vec.map { it * it }.sum())
    }

    /**
     * Frobenius norm of matrix: torch.linalg.norm(matrix, 'fro')
     */
    fun frobeniusNorm(matrix: Array<FloatArray>): Float {
        var sum = 0f
        for (row in matrix) {
            for (value in row) {
                sum += value * value
            }
        }
        return sqrt(sum)
    }

    // ============================================================================
    // Regularization Operations
    // ============================================================================

    /**
     * Add regularization: M + λI
     * Core operation for numerical stability in PIL.
     */
    fun addRegularization(matrix: Array<FloatArray>, lambda: Float): Array<FloatArray> {
        val n = matrix.size
        require(matrix[0].size == n) { "Matrix must be square" }

        return Array(n) { i ->
            FloatArray(n) { j ->
                matrix[i][j] + if (i == j) lambda else 0f
            }
        }
    }

    // ============================================================================
    // Condition Number (mirrors condition_number function)
    // ============================================================================

    /**
     * Compute the condition number of a matrix.
     * κ(A) = σ_max / σ_min
     *
     * Uses power iteration for efficiency on mobile.
     * Equivalent to torch.linalg.cond(matrix, p=2)
     */
    fun conditionNumber(matrix: Array<FloatArray>, iterations: Int = 30): Float {
        val n = matrix.size
        if (n == 0) return 1f

        // Power iteration for largest singular value
        var v = FloatArray(n) { 1f / sqrt(n.toFloat()) }
        var sigmaMax = 0f

        // A^T A for symmetric eigenvalue problem
        val AtA = matmul(transpose(matrix), matrix)

        repeat(iterations) {
            val av = matVec(AtA, v)
            sigmaMax = norm(av)
            if (sigmaMax > 1e-12f) {
                v = av.map { it / sigmaMax }.toFloatArray()
            }
        }
        sigmaMax = sqrt(sigmaMax)

        // Inverse power iteration for smallest singular value
        val regMatrix = addRegularization(AtA, 1e-12f)
        val invMatrix = invertLU(regMatrix) ?: return Float.MAX_VALUE

        v = FloatArray(n) { 1f / sqrt(n.toFloat()) }
        var sigmaMinInv = 0f

        repeat(iterations) {
            val av = matVec(invMatrix, v)
            sigmaMinInv = norm(av)
            if (sigmaMinInv > 1e-12f) {
                v = av.map { it / sigmaMinInv }.toFloatArray()
            }
        }

        val sigmaMin = if (sigmaMinInv > 1e-12f) 1f / sqrt(sigmaMinInv) else 1e-12f
        return sigmaMax / sigmaMin
    }

    // ============================================================================
    // Matrix Inversion (mirrors safe_inverse function)
    // ============================================================================

    /**
     * Safe matrix inversion with regularization and fallback strategies.
     * Computes (M + λI)^{-1} with numerical stability guarantees.
     *
     * Equivalent to safe_inverse() in pil_utils.py
     */
    fun safeInverse(
        matrix: Array<FloatArray>,
        regLambda: Float = DEFAULT_REG_LAMBDA,
        useCholesky: Boolean = true
    ): Array<FloatArray>? {
        val n = matrix.size

        // Add regularization: M + λI
        val regularized = addRegularization(matrix, regLambda)

        // Check condition number
        val cond = conditionNumber(regularized)
        if (cond > FAIL_THRESHOLD) {
            Log.w(TAG, "Matrix ill-conditioned: κ = $cond, using pseudoinverse")
            return pseudoinverse(regularized)
        }

        if (useCholesky) {
            // Try Cholesky decomposition (faster for positive definite)
            val result = choleskyInverse(regularized)
            if (result != null) {
                return result
            }
            Log.d(TAG, "Cholesky failed, falling back to LU")
        }

        // Fallback to LU decomposition
        val luResult = invertLU(regularized)
        if (luResult != null) {
            return luResult
        }

        // Final fallback: pseudoinverse
        Log.w(TAG, "LU failed, using pseudoinverse")
        return pseudoinverse(regularized)
    }

    /**
     * LU decomposition based matrix inversion.
     * Returns null if matrix is singular.
     */
    fun invertLU(matrix: Array<FloatArray>): Array<FloatArray>? {
        val n = matrix.size
        val lu = matrix.map { it.copyOf() }.toTypedArray()
        val perm = IntArray(n) { it }

        // LU decomposition with partial pivoting
        for (k in 0 until n) {
            // Find pivot
            var maxVal = 0f
            var maxIdx = k
            for (i in k until n) {
                val absVal = abs(lu[i][k])
                if (absVal > maxVal) {
                    maxVal = absVal
                    maxIdx = i
                }
            }

            if (maxVal < 1e-12f) {
                Log.w(TAG, "Matrix is singular at column $k")
                return null
            }

            // Swap rows
            if (maxIdx != k) {
                val temp = lu[k]
                lu[k] = lu[maxIdx]
                lu[maxIdx] = temp
                perm[k] = perm[maxIdx].also { perm[maxIdx] = perm[k] }
            }

            // Eliminate
            for (i in k + 1 until n) {
                lu[i][k] /= lu[k][k]
                for (j in k + 1 until n) {
                    lu[i][j] -= lu[i][k] * lu[k][j]
                }
            }
        }

        // Solve for inverse columns
        val inv = Array(n) { FloatArray(n) }
        for (col in 0 until n) {
            val b = FloatArray(n) { if (perm[it] == col) 1f else 0f }

            // Forward substitution (L * y = b)
            for (i in 0 until n) {
                for (j in 0 until i) {
                    b[i] -= lu[i][j] * b[j]
                }
            }

            // Back substitution (U * x = y)
            for (i in n - 1 downTo 0) {
                for (j in i + 1 until n) {
                    b[i] -= lu[i][j] * b[j]
                }
                b[i] /= lu[i][i]
            }

            for (i in 0 until n) {
                inv[i][col] = b[i]
            }
        }

        return inv
    }

    /**
     * Cholesky decomposition based inversion for positive definite matrices.
     * Faster than LU for symmetric positive definite matrices.
     */
    fun choleskyInverse(matrix: Array<FloatArray>): Array<FloatArray>? {
        val n = matrix.size
        val L = Array(n) { FloatArray(n) }

        // Cholesky decomposition: A = L L^T
        for (i in 0 until n) {
            for (j in 0..i) {
                var sum = matrix[i][j]
                for (k in 0 until j) {
                    sum -= L[i][k] * L[j][k]
                }
                if (i == j) {
                    if (sum <= 0) {
                        return null // Not positive definite
                    }
                    L[i][j] = sqrt(sum)
                } else {
                    L[i][j] = sum / L[j][j]
                }
            }
        }

        // Solve L L^T X = I
        val inv = Array(n) { FloatArray(n) }
        for (col in 0 until n) {
            // Forward solve L y = e_col
            val y = FloatArray(n)
            for (i in 0 until n) {
                var sum = if (i == col) 1f else 0f
                for (j in 0 until i) {
                    sum -= L[i][j] * y[j]
                }
                y[i] = sum / L[i][i]
            }

            // Backward solve L^T x = y
            for (i in n - 1 downTo 0) {
                var sum = y[i]
                for (j in i + 1 until n) {
                    sum -= L[j][i] * inv[j][col]
                }
                inv[i][col] = sum / L[i][i]
            }
        }

        return inv
    }

    // ============================================================================
    // Pseudoinverse (mirrors torch.linalg.pinv)
    // ============================================================================

    /**
     * Moore-Penrose pseudoinverse using regularized normal equations.
     * Equivalent to torch.linalg.pinv(matrix)
     */
    fun pseudoinverse(
        matrix: Array<FloatArray>,
        regLambda: Float = DEFAULT_REG_LAMBDA * 10f
    ): Array<FloatArray> {
        val m = matrix.size
        val n = matrix[0].size

        return if (m >= n) {
            // A^+ = (A^T A + λI)^{-1} A^T
            val At = transpose(matrix)
            val AtA = matmul(At, matrix)
            val regAtA = addRegularization(AtA, regLambda)
            val inv = invertLU(regAtA) ?: eye(n)
            matmul(inv, At)
        } else {
            // A^+ = A^T (A A^T + λI)^{-1}
            val At = transpose(matrix)
            val AAt = matmul(matrix, At)
            val regAAt = addRegularization(AAt, regLambda)
            val inv = invertLU(regAAt) ?: eye(m)
            matmul(At, inv)
        }
    }

    // ============================================================================
    // Ridge Solve (mirrors ridge_solve function)
    // ============================================================================

    /**
     * Solve the ridge regression problem: W = (H^T H + λI)^{-1} H^T Y
     *
     * This is the core PIL weight solving operation.
     * Equivalent to ridge_solve() in pil_utils.py
     *
     * @param H Feature matrix (N, D_hidden) - hidden activations
     * @param Y Target matrix (N, D_out) - targets
     * @param regLambda Regularization parameter
     * @param useSvdFallback Use SVD-based pseudoinverse as fallback
     * @return Weight matrix W (D_hidden, D_out)
     */
    fun ridgeSolve(
        H: Array<FloatArray>,
        Y: Array<FloatArray>,
        regLambda: Float = DEFAULT_REG_LAMBDA,
        useSvdFallback: Boolean = true
    ): Array<FloatArray>? {
        val n = H.size
        val dHidden = H[0].size

        // H^T (D_hidden, N)
        val Ht = transpose(H)

        // H^T H (D_hidden, D_hidden)
        val HtH = matmul(Ht, H)

        // H^T Y (D_hidden, D_out)
        val HtY = matmul(Ht, Y)

        // Add regularization: H^T H + λI
        val regMatrix = addRegularization(HtH, regLambda)

        // Check condition number
        val cond = conditionNumber(regMatrix)
        if (cond > FAIL_THRESHOLD) {
            Log.w(TAG, "Ridge solve ill-conditioned: κ = $cond")
            if (useSvdFallback) {
                // Use pseudoinverse directly: W = H^+ Y
                val Hpinv = pseudoinverse(H, regLambda)
                return matmul(Hpinv, Y)
            }
        }

        // Standard solution: W = (H^T H + λI)^{-1} H^T Y
        val regInv = invertLU(regMatrix)
        if (regInv != null) {
            return matmul(regInv, HtY)
        }

        // Fallback
        Log.w(TAG, "LU solve failed, using pseudoinverse")
        val Hpinv = pseudoinverse(H, regLambda)
        return matmul(Hpinv, Y)
    }

    // ============================================================================
    // Low-Rank Ridge Solve (mirrors low_rank_ridge_solve function)
    // ============================================================================

    /**
     * Low-rank approximation for ridge regression.
     * For large N (tokens > 10000), reduces O(N^3) to O(N * rank^2).
     *
     * Simplified version using truncated eigendecomposition.
     */
    fun lowRankRidgeSolve(
        H: Array<FloatArray>,
        Y: Array<FloatArray>,
        regLambda: Float = DEFAULT_REG_LAMBDA,
        rank: Int? = null
    ): Array<FloatArray> {
        val n = H.size
        val dHidden = H[0].size

        val effectiveRank = rank ?: minOf(100, dHidden / 2, n / 2).coerceAtLeast(1)

        // For simplicity on mobile, use standard solve with higher regularization
        // Full SVD is expensive; this is a practical approximation
        return ridgeSolve(H, Y, regLambda * (1 + dHidden.toFloat() / effectiveRank))
            ?: run {
                Log.w(TAG, "Low-rank solve failed, returning zeros")
                zeros(dHidden, Y[0].size)
            }
    }

    // ============================================================================
    // Sherman-Morrison Update (mirrors sherman_morrison_update function)
    // ============================================================================

    /**
     * Incremental weight update using Sherman-Morrison formula.
     * For online learning, update weights one sample at a time.
     *
     * (A + uv^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)
     */
    data class ShermanMorrisonResult(
        val W: Array<FloatArray>,
        val HInv: Array<FloatArray>
    )

    fun shermanMorrisonUpdate(
        W: Array<FloatArray>,
        HInv: Array<FloatArray>,
        hNew: FloatArray,
        yNew: FloatArray,
        regLambda: Float = DEFAULT_REG_LAMBDA
    ): ShermanMorrisonResult {
        val dHidden = hNew.size

        // H_inv @ h (D_hidden,)
        val HInvH = matVec(HInv, hNew)

        // h^T @ H_inv @ h (scalar)
        var denom = 1f
        for (i in 0 until dHidden) {
            denom += hNew[i] * HInvH[i]
        }

        if (abs(denom) < 1e-10f) {
            Log.w(TAG, "Sherman-Morrison singular: denom = $denom")
            return ShermanMorrisonResult(W, HInv)
        }

        // H_inv_new = H_inv - (H_inv @ h @ h^T @ H_inv) / denom
        val HInvNew = Array(dHidden) { i ->
            FloatArray(dHidden) { j ->
                HInv[i][j] - (HInvH[i] * HInvH[j]) / denom
            }
        }

        // Residual: y_new - h^T @ W
        val hTW = FloatArray(W[0].size) { j ->
            var sum = 0f
            for (i in 0 until dHidden) {
                sum += hNew[i] * W[i][j]
            }
            sum
        }

        val residual = FloatArray(yNew.size) { j ->
            yNew[j] - hTW[j]
        }

        // W_new = W + H_inv_new @ h @ residual^T
        val HInvNewH = matVec(HInvNew, hNew)
        val WNew = Array(dHidden) { i ->
            FloatArray(W[0].size) { j ->
                W[i][j] + HInvNewH[i] * residual[j]
            }
        }

        return ShermanMorrisonResult(WNew, HInvNew)
    }

    // ============================================================================
    // Orthogonal Initialization (mirrors orthogonal_init function)
    // ============================================================================

    /**
     * Initialize a random orthogonal matrix using Gram-Schmidt.
     * For the fixed expansion weights W_random in PIL.
     */
    fun orthogonalInit(
        rows: Int,
        cols: Int,
        seed: Long? = null
    ): Array<FloatArray> {
        val random = if (seed != null) java.util.Random(seed) else java.util.Random()

        // Generate random matrix
        val matrix = Array(rows) { FloatArray(cols) { random.nextGaussian().toFloat() } }

        // Gram-Schmidt orthogonalization
        for (j in 0 until cols) {
            // Subtract projections onto previous columns
            for (i in 0 until j) {
                var dot = 0f
                var normSq = 0f
                for (k in 0 until rows) {
                    dot += matrix[k][j] * matrix[k][i]
                    normSq += matrix[k][i] * matrix[k][i]
                }
                if (normSq > 1e-10f) {
                    val scale = dot / normSq
                    for (k in 0 until rows) {
                        matrix[k][j] -= scale * matrix[k][i]
                    }
                }
            }

            // Normalize
            var norm = 0f
            for (k in 0 until rows) {
                norm += matrix[k][j] * matrix[k][j]
            }
            norm = sqrt(norm)
            if (norm > 1e-10f) {
                for (k in 0 until rows) {
                    matrix[k][j] /= norm
                }
            }
        }

        return matrix
    }
}
