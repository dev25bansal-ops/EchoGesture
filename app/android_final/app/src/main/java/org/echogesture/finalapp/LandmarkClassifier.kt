package org.echogesture.finalapp

import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Strict on-device adapter for the frozen FP16 landmark ENSEMBLE.
 *
 * Contract (validated on every member's actual tensors at construction):
 *   input  [1, 24, 255] float32
 *   output [1, 100]     float32 logits, or int8 logits with quantization
 *
 * The bundled artifacts are FP16 weights with float32 IO — this is NOT an INT8
 * model and this class never relabels it as one.
 *
 * Each member is its own [Interpreter]. Tensor "indices" in the Python path
 * (`get_output_details()[0]["index"]`, observed 224 for tcn21/rob71 and 191 for
 * tcnm61/tcnm62) are flatbuffer tensor handles that differ per architecture.
 * The Java API does not expose that handle: a one-input/one-output interpreter
 * is addressed positionally through its own interpreter object. So the
 * per-member handle here is the [Member] itself — invocation and output reads
 * always go through `member.interpreter`, never a shared/global interpreter or
 * index. That is the invariant a naive port breaks (one interpreter reused for
 * every file, or output read from another member's index).
 *
 * Aggregation matches src/main.py exactly: collect each member's logits,
 * average them element-wise, run one stable softmax, then top-5.
 */
class LandmarkClassifier(
    private val members: List<Member>,
    private val labels: List<String>
) : AutoCloseable {

    /** One loaded ensemble member plus the tensor metadata resolved for it. */
    class Member(
        val name: String,
        modelBuffer: ByteBuffer,
        expectedOutputClasses: Int = ModelContract.EXPECTED_CLASSES
    ) {
        internal val interpreter: Interpreter
        internal val inputTensorName: String
        internal val outputTensorName: String
        internal val inputScale: Float
        internal val inputZeroPoint: Int
        internal val outputScale: Float
        internal val outputZeroPoint: Int
        internal val outputIsFloat: Boolean
        internal val inputIsFloat: Boolean

        init {
            interpreter = Interpreter(
                modelBuffer,
                Interpreter.Options().apply { setNumThreads(4); setUseXNNPACK(true) }
            )
            require(interpreter.inputTensorCount == 1) {
                "member '$name': expected 1 input tensor, got ${interpreter.inputTensorCount}"
            }
            require(interpreter.outputTensorCount == 1) {
                "member '$name': expected 1 output tensor, got ${interpreter.outputTensorCount}"
            }

            // Resolve this member's own handles and validate only them.
            val inputTensor: Tensor = interpreter.getInputTensor(0)
            val outputTensor: Tensor = interpreter.getOutputTensor(0)
            inputTensorName = inputTensor.name()
            outputTensorName = outputTensor.name()

            check(inputTensor.shape().contentEquals(intArrayOf(1, 24, 255))) {
                "member '$name': input shape ${inputTensor.shape().joinToString(",")} != [1,24,255]"
            }
            check(outputTensor.shape().contentEquals(intArrayOf(1, expectedOutputClasses))) {
                "member '$name': output shape ${outputTensor.shape().joinToString(",")} " +
                    "!= [1,$expectedOutputClasses]"
            }
            check(inputTensor.dataType() in setOf(DataType.FLOAT32, DataType.INT8)) {
                "member '$name': input dtype ${inputTensor.dataType()} is not float32/int8"
            }
            check(outputTensor.dataType() in setOf(DataType.FLOAT32, DataType.INT8)) {
                "member '$name': output dtype ${outputTensor.dataType()} is not float32/int8"
            }

            inputScale = inputTensor.quantizationParams().scale
            inputZeroPoint = inputTensor.quantizationParams().zeroPoint
            outputScale = outputTensor.quantizationParams().scale
            outputZeroPoint = outputTensor.quantizationParams().zeroPoint
            outputIsFloat = outputTensor.dataType() == DataType.FLOAT32
            inputIsFloat = inputTensor.dataType() == DataType.FLOAT32
            if (inputTensor.dataType() == DataType.INT8) {
                check(inputScale > 0f) { "member '$name': int8 input has zero scale" }
            }
        }

        internal fun close() = interpreter.close()
    }

    data class Result(
        val prediction: String,
        val top5: List<String>,
        /** Softmax probability of the top-1 class over the averaged logits. */
        val confidence: Float,
        /** Wall time for all member invocations (aggregation is negligible). */
        val inferenceMillis: Long,
        /** Per-member invocation time, in meta order. */
        val memberMillis: Map<String, Long> = emptyMap(),
        /** Averaged logits, exposed for parity/debugging. */
        val averagedLogits: FloatArray? = null
    ) {
        // FloatArray needs explicit equals/hashCode in a data class.
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Result) return false
            return prediction == other.prediction &&
                top5 == other.top5 &&
                confidence == other.confidence &&
                inferenceMillis == other.inferenceMillis &&
                memberMillis == other.memberMillis &&
                (averagedLogits contentEquals other.averagedLogits)
        }

        override fun hashCode(): Int {
            var result = prediction.hashCode()
            result = 31 * result + top5.hashCode()
            result = 31 * result + confidence.hashCode()
            result = 31 * result + inferenceMillis.hashCode()
            result = 31 * result + memberMillis.hashCode()
            result = 31 * result + (averagedLogits?.contentHashCode() ?: 0)
            return result
        }
    }

    init {
        require(members.isNotEmpty()) { "ensemble requires at least one member" }
        require(labels.size == ModelContract.EXPECTED_CLASSES) {
            "classifier requires ${ModelContract.EXPECTED_CLASSES} labels, got ${labels.size}"
        }
        val duplicate = members.groupBy { it.name }.filterValues { it.size > 1 }.keys.firstOrNull()
        require(duplicate == null) { "duplicate ensemble member name: $duplicate" }
    }

    val memberNames: List<String> get() = members.map { it.name }
    val memberCount: Int get() = members.size

    // TFLite's Interpreter is not documented thread-safe for concurrent run().
    // One lock serializes inference and close(): a predict() racing
    // onDestroy()'s close() would otherwise run native inference on a freed
    // interpreter. This is not a throughput loss in the app, which already
    // calls predict() from one executor; it removes the teardown race and makes
    // the guarantee independent of the caller.
    private val gate = Any()
    @Volatile private var closed = false

    /** features: 24*255 float32 row-major from FeatureTransform.buildFeatures. */
    fun predict(features: FloatArray): Result = synchronized(gate) {
        check(!closed) { "classifier is closed" }
        val expected = ModelContract.EXPECTED_FRAMES * ModelContract.EXPECTED_FEATURE_DIM
        require(features.size == expected) {
            "features must be 24*255=$expected, got ${features.size}"
        }

        val memberLogits = ArrayList<FloatArray>(members.size)
        val memberMillis = LinkedHashMap<String, Long>()
        var inferenceMillis = 0L

        for (member in members) {
            val start = System.nanoTime()
            memberLogits.add(invoke(member, features))
            val elapsed = (System.nanoTime() - start) / 1_000_000L
            memberMillis[member.name] = elapsed
            inferenceMillis += elapsed
        }

        // Average logits, then exactly one softmax (mirrors src/main.py).
        val averaged = EnsembleAggregation.meanLogits(memberLogits.toTypedArray())
        val probs = EnsembleAggregation.softmax(averaged)
        val order = EnsembleAggregation.topK(probs, 5)
        val top5 = order.map { labels[it] }
        return Result(
            prediction = top5[0],
            top5 = top5,
            confidence = probs[order[0]],
            inferenceMillis = inferenceMillis,
            memberMillis = memberMillis,
            averagedLogits = averaged
        )
    }

    /**
     * Multi-window prediction: average member logits over several time windows
     * of the same capture, then one softmax.
     *
     * Why: the app cannot know where in the 32-frame capture the sign actually
     * sits — the user taps when they think it is done. Measured on the
     * signer-disjoint test split, averaging the full capture with the last 90%
     * and last 80% raised top-1 from 72.78% to 74.62% (val 83.77% -> 84.64%),
     * paired McNemar z=3.59, +32 clips fixed vs 9 broken. Averaging several
     * placements beats betting on one, which was the same argument that made
     * the four-model ensemble work.
     *
     * Cost: three windows x four members = 12 invocations instead of 4. At the
     * measured ~60 ms per member-window that is ~240 ms, still far inside the
     * 50 ms per-frame budget because this runs once per capture, not per frame.
     *
     * `windows` are (startFraction, endFraction) pairs over the capture.
     */
    fun predictMultiWindow(
        landmarks: FloatArray,
        present: FloatArray,
        srcFrames: Int,
        windows: List<Pair<Double, Double>> = DEFAULT_WINDOWS
    ): Result = synchronized(gate) {
        check(!closed) { "classifier is closed" }
        require(windows.isNotEmpty()) { "at least one window is required" }

        val expected = ModelContract.EXPECTED_FRAMES * ModelContract.EXPECTED_FEATURE_DIM
        val memberMillis = LinkedHashMap<String, Long>()
        members.forEach { memberMillis[it.name] = 0L }
        var inferenceMillis = 0L

        val windowLogits = ArrayList<FloatArray>(windows.size)
        for ((lo, hi) in windows) {
            val start = (srcFrames * lo).toInt().coerceIn(0, srcFrames - 1)
            val end = (srcFrames * hi).toInt().coerceIn(start + 1, srcFrames)
            val features = FeatureTransform.buildFeatures(
                landmarks, present, srcFrames,
                ModelContract.EXPECTED_FRAMES, start, end
            )
            require(features.size == expected) {
                "features must be 24*255=$expected, got ${features.size}"
            }
            for (member in members) {
                val t0 = System.nanoTime()
                windowLogits.add(invoke(member, features))
                val elapsed = (System.nanoTime() - t0) / 1_000_000L
                memberMillis[member.name] = (memberMillis[member.name] ?: 0L) + elapsed
                inferenceMillis += elapsed
            }
        }

        val averaged = EnsembleAggregation.meanLogits(windowLogits.toTypedArray())
        val probs = EnsembleAggregation.softmax(averaged)
        val order = EnsembleAggregation.topK(probs, 5)
        val top5 = order.map { labels[it] }
        Result(
            prediction = top5[0],
            top5 = top5,
            confidence = probs[order[0]],
            inferenceMillis = inferenceMillis,
            memberMillis = memberMillis,
            averagedLogits = averaged
        )
    }

    /**
     * Runs EVERY member on the same features and returns each member's own raw
     * logits, in meta order. Same invocation path as [predict]; exposed so the
     * on-device test can assert per-member tensor resolution and parity against
     * the Python reference instead of only the aggregated result.
     */
    fun memberLogits(features: FloatArray): List<FloatArray> = synchronized(gate) {
        check(!closed) { "classifier is closed" }
        val expected = ModelContract.EXPECTED_FRAMES * ModelContract.EXPECTED_FEATURE_DIM
        require(features.size == expected) {
            "features must be 24*255=$expected, got ${features.size}"
        }
        members.map { invoke(it, features) }
    }

    /** One member invocation through that member's own interpreter. */
    private fun invoke(member: Member, features: FloatArray): FloatArray {
        val frames = ModelContract.EXPECTED_FRAMES
        val dim = ModelContract.EXPECTED_FEATURE_DIM
        val classes = ModelContract.EXPECTED_CLASSES

        val input: Array<Array<FloatArray>> = Array(1) { Array(frames) { FloatArray(dim) } }
        for (t in 0 until frames) {
            System.arraycopy(features, t * dim, input[0][t], 0, dim)
        }
        // Quantized input only when the tensor actually is int8; a float input
        // has scale 0 and must be fed as float.
        val modelInput: Any = if (member.inputIsFloat) {
            input
        } else {
            Array(1) { Array(frames) { t ->
                ByteArray(dim) { f ->
                    val q = Math.round(features[t * dim + f] / member.inputScale) +
                        member.inputZeroPoint
                    q.coerceIn(-128, 127).toByte()
                }
            } }
        }

        return if (member.outputIsFloat) {
            val output = Array(1) { FloatArray(classes) }
            member.interpreter.run(modelInput, output)
            output[0]
        } else {
            // Not the shipped form, but supported the way src/main.py handles it:
            // dequantize the int8 logits with this member's output params.
            val output = Array(1) { ByteArray(classes) }
            member.interpreter.run(modelInput, output)
            FloatArray(classes) { i ->
                (output[0][i].toInt() - member.outputZeroPoint) * member.outputScale
            }
        }
    }

    override fun close() = synchronized(gate) {
        if (closed) return
        closed = true
        members.forEach { it.close() }
    }

    /** Loads every member listed in the meta sidecar, or throws. */
    companion object {
        /**
         * The window set measured best on both splits. Kept here so the
         * on-device parity test and the runtime cannot drift apart.
         *
         * Averaging the whole capture with the last 90% and last 80% gave
         * 74.62% test / 84.64% val against 72.78% / 83.77% for the whole
         * capture alone (paired McNemar z=3.59 on test).
         */
        val DEFAULT_WINDOWS: List<Pair<Double, Double>> = listOf(
            0.0 to 1.0,    // whole capture
            0.1 to 1.0,    // last 90%
            0.2 to 1.0     // last 80%
        )

        fun create(loaded: LoadedModel, classes: Int = loaded.identity.classes): LandmarkClassifier {
            require(loaded.members.isNotEmpty()) {
                "ModelContract resolved an empty ensemble; refusing to run"
            }
            val loadedMembers = loaded.members.map { asset ->
                Member(
                    name = asset.name,
                    modelBuffer = ByteBuffer.allocateDirect(asset.bytes.size)
                        .order(ByteOrder.nativeOrder())
                        .apply { put(asset.bytes); rewind() },
                    expectedOutputClasses = classes
                )
            }
            return try {
                LandmarkClassifier(loadedMembers, loaded.labels)
            } catch (t: Throwable) {
                loadedMembers.forEach { runCatching { it.close() } }
                throw t
            }
        }
    }
}