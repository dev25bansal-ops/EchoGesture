package org.echogesture.finalapp

import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Strict on-device adapter for the frozen FP16 landmark classifier.
 *
 * Contract (validated on the actual tensors at construction):
 *   input  [1, 24, 255] float32
 *   output [1, 100]     float32 logits
 *
 * The bundled artifact is FP16 weights with float32 IO — it is NOT an INT8
 * model, and this class never relabels it as one. Output is the same stable
 * softmax + top-5 as src/main.py, returning raw logits decoded probabilities.
 */
class LandmarkClassifier(
    private val modelBuffer: ByteBuffer,
    private val labels: List<String>
) : AutoCloseable {

    private val interpreter: Interpreter
    private val inputTensor: Tensor
    private val outputTensor: Tensor

    init {
        require(labels.size == 100) { "classifier requires 100 labels, got ${labels.size}" }
        interpreter = Interpreter(
            modelBuffer,
            Interpreter.Options().apply { setNumThreads(4); setUseXNNPACK(true) }
        )
        require(interpreter.getInputTensorCount() == 1) { "expected 1 input tensor" }
        require(interpreter.getOutputTensorCount() == 1) { "expected 1 output tensor" }
        inputTensor = interpreter.getInputTensor(0)
        outputTensor = interpreter.getOutputTensor(0)

        check(inputTensor.dataType() == DataType.FLOAT32) {
            "input dtype ${inputTensor.dataType()} is not float32"
        }
        check(outputTensor.dataType() == DataType.FLOAT32) {
            "output dtype ${outputTensor.dataType()} is not float32"
        }
        check(inputTensor.shape().contentEquals(intArrayOf(1, 24, 255))) {
            "input shape ${inputTensor.shape().joinToString(",")} != [1,24,255]"
        }
        check(outputTensor.shape().contentEquals(intArrayOf(1, 100))) {
            "output shape ${outputTensor.shape().joinToString(",")} != [1,100]"
        }
    }

    data class Result(
        val prediction: String,
        val top5: List<String>,
        val confidence: Float,
        val inferenceMillis: Long
    )

    /** features: 24*255 float32 row-major from FeatureTransform.buildFeatures. */
    fun predict(features: FloatArray): Result {
        require(features.size == 24 * 255) {
            "features must be 24*255=${24 * 255}, got ${features.size}"
        }

        val input = Array(1) { Array(24) { FloatArray(255) } }
        for (t in 0 until 24) System.arraycopy(features, t * 255, input[0][t], 0, 255)
        val output = Array(1) { FloatArray(100) }

        val start = System.nanoTime()
        interpreter.run(input, output)
        val inferenceMillis = (System.nanoTime() - start) / 1_000_000L

        val logits = output[0]
        val mx = logits.maxOrNull() ?: 0f
        var sum = 0.0
        val probs = FloatArray(100) { i ->
            val e = kotlin.math.exp((logits[i] - mx).toDouble()).toFloat()
            sum += e
            e
        }
        for (i in probs.indices) probs[i] = (probs[i] / sum.toFloat())

        val order = probs.indices.sortedByDescending { probs[it] }
        val top5 = order.take(5).map { labels[it] }
        return Result(top5[0], top5, probs[order[0]], inferenceMillis)
    }

    override fun close() = interpreter.close()
}