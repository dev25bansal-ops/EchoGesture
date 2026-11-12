package org.echogesture.finalapp

import kotlin.math.exp

/**
 * Pure ensemble math, kept free of Android/TFLite types so JVM unit tests can
 * pin it against the Python reference.
 *
 * Reference semantics (src/main.py::_run_ensemble + predict):
 *   1. every member produces raw float32 logits for the same class axis,
 *   2. logits are averaged element-wise over members,
 *   3. exactly one stable softmax runs over the averaged logits,
 *   4. top-5 is taken from that single probability vector.
 */
internal object EnsembleAggregation {

    /** Element-wise mean of member logits; the softmax happens afterwards. */
    fun meanLogits(memberLogits: Array<FloatArray>): FloatArray {
        require(memberLogits.isNotEmpty()) { "ensemble requires at least one member" }
        val classes = memberLogits[0].size
        require(classes > 0) { "member logits must be non-empty" }
        val mean = FloatArray(classes)
        for ((position, logits) in memberLogits.withIndex()) {
            require(logits.size == classes) {
                "member $position produced ${logits.size} logits, expected $classes"
            }
            for (i in 0 until classes) mean[i] += logits[i]
        }
        val count = memberLogits.size.toFloat()
        for (i in 0 until classes) mean[i] /= count
        return mean
    }

    /** Numerically stable softmax, matching numpy exp(x - max) / sum. */
    fun softmax(logits: FloatArray): FloatArray {
        require(logits.isNotEmpty()) { "softmax needs at least one logit" }
        val max = logits.max()
        var sum = 0.0
        val probs = FloatArray(logits.size) { i ->
            val e = exp((logits[i] - max).toDouble())
            sum += e
            e.toFloat()
        }
        require(sum > 0.0) { "softmax normalizer is zero" }
        for (i in probs.indices) probs[i] = (probs[i] / sum).toFloat()
        return probs
    }

    /** Descending probability order, first [k] indices (ties keep index order). */
    fun topK(probs: FloatArray, k: Int): IntArray {
        require(k in 1..probs.size) { "topK=$k outside 1..${probs.size}" }
        return probs.indices.sortedByDescending { probs[it] }.take(k).toIntArray()
    }
}