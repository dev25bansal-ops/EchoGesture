package org.echogesture.v7

import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Thin TFLite adapter for the pinned FP16 candidate. Contains NO graph math
 * beyond the same stable softmax + top-5 used by src/main.py. Contract is
 * validated at construction: input [1,24,255] float32, output [1,100] float32.
 * The bundled artifact is FP16 weights with float32 IO -- it is NOT an INT8
 * model, and this class never relabels it as one.
 */
class Classifier(model: MappedByteBuffer) : AutoCloseable {
    companion object {
        fun fromAsset(path: String): Classifier {
            FileInputStream(path).use { fis ->
                val mbb: MappedByteBuffer = fis.channel.map(
                    FileChannel.MapMode.READ_ONLY, 0, fis.channel.size())
                return Classifier(mbb)
            }
        }
    }

    private val interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })
    private val inputShape: IntArray = interpreter.getInputTensor(0).shape()
    private val outputShape: IntArray = interpreter.getOutputTensor(0).shape()
    val labels: List<String>

    init {
        require(inputShape.contentEquals(intArrayOf(1, 24, 255))) { "input shape ${inputShape.joinToString()} != [1,24,255]" }
        require(outputShape.contentEquals(intArrayOf(1, 100))) { "output shape ${outputShape.joinToString()} != [1,100]" }
        labels = emptyList() // populated via loadLabels below
    }

    /** labels.json is a JSON array of 100 uppercase glosses, index = class id. */
    fun loadLabels(json: String): List<String> {
        val stripped = json.trim().removePrefix("[").removeSuffix("]")
        return stripped.split(",").map { it.trim().trim('"') }.also {
            require(it.size == 100) { "expected 100 labels, got ${it.size}" }
        }
    }

    data class Result(val prediction: String, val top5: List<String>, val confidence: Float)

    /** features: 24*255 float32 row-major, from FeatureTransform.buildFeatures. */
    fun predict(features: FloatArray, vocab: List<String>): Result {
        require(features.size == 24 * 255) { "features must be 24*255=${24 * 255}, got ${features.size}" }
        val input = Array(1) { Array(24) { FloatArray(255) } }
        for (t in 0 until 24) System.arraycopy(features, t * 255, input[0][t], 0, 255)
        val output = Array(1) { FloatArray(100) }
        interpreter.run(input, output)
        val logits = output[0]
        // stable softmax (same as src/main.py)
        val mx = logits.maxOrNull()!!
        var sum = 0.0
        val probs = FloatArray(100) { i -> kotlin.math.exp((logits[i] - mx).toDouble()).toFloat().also { sum += it } }
        for (i in probs.indices) probs[i] = (probs[i] / sum).toFloat()
        val order = probs.indices.sortedByDescending { probs[it] }
        val top5 = order.take(5).map { vocab[it] }
        return Result(top5[0], top5, probs[order[0]])
    }

    override fun close() = interpreter.close()
}
