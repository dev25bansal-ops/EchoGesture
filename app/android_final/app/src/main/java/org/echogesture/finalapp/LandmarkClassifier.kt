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
        expectedOutputClasses: Int = ModelContract.EXPECTED_CLASSES,
        /** True when this member takes 381-wide x,y,z features. */
        val useZ: Boolean = false
    ) {
        internal val interpreter: Interpreter
        internal val featureTensorName: String
        internal val maskTensorName: String
        internal val featureTensorIndex: Int
        internal val maskTensorIndex: Int
        internal val featureDim: Int
        internal val outputTensorName: String
        internal val inputScale: Float
        internal val inputZeroPoint: Int
        internal val maskScale: Float
        internal val maskZeroPoint: Int
        internal val maskIsFloat: Boolean
        internal val outputScale: Float
        internal val outputZeroPoint: Int
        internal val outputIsFloat: Boolean
        internal val inputIsFloat: Boolean

        init {
            interpreter = Interpreter(
                modelBuffer,
                Interpreter.Options().apply { setNumThreads(4); setUseXNNPACK(true) }
            )
            // The masked models take TWO inputs. Resolve them BY NAME: the
            // converter reports the mask as input 0, so positional code feeds
            // the mask where the features belong and fails only as a shape
            // mismatch several layers deeper.
            require(interpreter.inputTensorCount == 2) {
                "member '$name': expected 2 inputs (features + mask), " +
                    "got ${interpreter.inputTensorCount}"
            }
            require(interpreter.outputTensorCount == 1) {
                "member '$name': expected 1 output tensor, got ${interpreter.outputTensorCount}"
            }

            // Resolve this member's own handles and validate only them.
            var fIdx = -1
            var mIdx = -1
            for (i in 0 until interpreter.inputTensorCount) {
                val n = interpreter.getInputTensor(i).name()
                if (n.contains("features")) fIdx = i
                if (n.contains("mask")) mIdx = i
            }
            check(fIdx >= 0) { "member '$name': no input tensor named like 'features'" }
            check(mIdx >= 0) { "member '$name': no input tensor named like 'mask'" }
            require(fIdx != mIdx) { "member '$name': features and mask resolved to the same tensor" }
            featureTensorIndex = fIdx
            maskTensorIndex = mIdx

            val inputTensor: Tensor = interpreter.getInputTensor(fIdx)
            val maskTensor: Tensor = interpreter.getInputTensor(mIdx)
            val outputTensor: Tensor = interpreter.getOutputTensor(0)
            featureTensorName = inputTensor.name()
            maskTensorName = maskTensor.name()
            outputTensorName = outputTensor.name()

            featureDim = FeatureTransform.featureDim(useZ)
            check(inputTensor.shape().contentEquals(intArrayOf(1, 24, featureDim))) {
                "member '$name': features shape ${inputTensor.shape().joinToString(",")} " +
                    "!= [1,24,$featureDim]"
            }
            check(maskTensor.shape().contentEquals(intArrayOf(1, 24))) {
                "member '$name': mask shape ${maskTensor.shape().joinToString(",")} != [1,24]"
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
            maskScale = maskTensor.quantizationParams().scale
            maskZeroPoint = maskTensor.quantizationParams().zeroPoint
            maskIsFloat = maskTensor.dataType() == DataType.FLOAT32
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
        /**
         * Top-1 minus top-2 probability.
         *
         * Measured on 82 authorized validation clips, this separates correct
         * from incorrect far better than the raw top-1 score does: mean margin
         * is 0.378 on correct predictions and 0.064 on incorrect ones. At a
         * 0.15 margin cut the pipeline keeps 61% of captures at 94% accuracy;
         * the 0.35 score cut keeps only 45% at 94.6%. The split-half check
         * repeats that on both halves. It is a ranking separation, not a
         * probability of correctness.
         */
        val margin: Float = 0f,
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

    /**
     * @param features the 255-wide x,y tensor (used by `use_z=false` members)
     * @param mask     the 24-step attention mask from the SAME landmark pass
     * @param featuresXYZ the 381-wide x,y,z tensor (used by `use_z=true` members)
     */
    fun predict(
        features: FloatArray,
        mask: FloatArray,
        featuresXYZ: FloatArray = FloatArray(0)
    ): Result = synchronized(gate) {
        check(!closed) { "classifier is closed" }
        validateInputs(features, mask, featuresXYZ)

        val memberLogits = ArrayList<FloatArray>(members.size)
        val memberMillis = LinkedHashMap<String, Long>()
        var inferenceMillis = 0L

        for (member in members) {
            val start = System.nanoTime()
            memberLogits.add(invoke(member, if (member.useZ) featuresXYZ else features, mask))
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
            margin = probs[order[0]] - probs[order[1]],
            inferenceMillis = inferenceMillis,
            memberMillis = memberMillis,
            averagedLogits = averaged
        )
    }

    /**
     * Runs EVERY member and returns each member's own raw logits, in meta
     * order. Same invocation path as [predict]; exposed so the on-device test
     * can assert per-member tensor resolution and parity against the Python
     * reference instead of only the aggregated result.
     */
    fun memberLogits(
        features: FloatArray,
        mask: FloatArray,
        featuresXYZ: FloatArray = FloatArray(0)
    ): List<FloatArray> = synchronized(gate) {
        check(!closed) { "classifier is closed" }
        validateInputs(features, mask, featuresXYZ)
        members.map { invoke(it, if (it.useZ) featuresXYZ else features, mask) }
    }

    private fun validateInputs(features: FloatArray, mask: FloatArray, featuresXYZ: FloatArray) {
        val frames = ModelContract.EXPECTED_FRAMES
        val xyExpected = frames * ModelContract.EXPECTED_FEATURE_DIM
        require(features.size == xyExpected) {
            "features must be 24*255=$xyExpected, got ${features.size}"
        }
        require(mask.size == frames) { "mask must be $frames values, got ${mask.size}" }
        val needsZ = members.any { it.useZ }
        if (needsZ) {
            val xyzExpected = frames * FeatureTransform.FEATURE_DIM_XYZ
            require(featuresXYZ.size == xyzExpected) {
                "ensemble has z-members but featuresXYZ is ${featuresXYZ.size}, expected $xyzExpected"
            }
        }
    }

    /**
     * One member invocation through that member's own interpreter.
     *
     * Two inputs, addressed BY TENSOR OBJECT — never by position. The
     * converter reports the mask first, so positional feeding would put the
     * mask where the features go. Building the map from each member's own
     * resolved tensors makes the ordering irrelevant.
     */
    private fun invoke(member: Member, features: FloatArray, mask: FloatArray): FloatArray {
        val frames = ModelContract.EXPECTED_FRAMES
        val dim = member.featureDim
        val classes = ModelContract.EXPECTED_CLASSES
        require(features.size == frames * dim) {
            "member '${member.name}': got ${features.size} features, expected ${frames * dim}"
        }

        val featureInput: Any = if (member.inputIsFloat) {
            Array(1) { Array(frames) { FloatArray(dim) } }.also { array ->
                for (t in 0 until frames) {
                    System.arraycopy(features, t * dim, array[0][t], 0, dim)
                }
            }
        } else {
            Array(1) { Array(frames) { t ->
                ByteArray(dim) { f ->
                    val q = Math.round(features[t * dim + f] / member.inputScale) +
                        member.inputZeroPoint
                    q.coerceIn(-128, 127).toByte()
                }
            } }
        }
        // Mask is float32 in the shipped artifacts; quantize only if it is not.
        val maskInput: Any = if (member.maskIsFloat) {
            Array(1) { mask.copyOf(frames) }
        } else {
            Array(1) { ByteArray(frames) { t ->
                val q = Math.round(mask[t] / member.maskScale) + member.maskZeroPoint
                q.coerceIn(-128, 127).toByte()
            } }
        }

        // Indexed by the member's own tensor indices, so the converter's
        // ordering is irrelevant.
        // Positional ARRAY, with each buffer placed at that member's own
        // resolved tensor index, so the converter's ordering cannot matter.
        val inputs = arrayOfNulls<Any>(member.interpreter.inputTensorCount)
        inputs[member.featureTensorIndex] = featureInput
        inputs[member.maskTensorIndex] = maskInput
        val outputs = HashMap<Int, Any>(1)

        return if (member.outputIsFloat) {
            val output = Array(1) { FloatArray(classes) }
            outputs[0] = output
            member.interpreter.runForMultipleInputsOutputs(inputs, outputs)
            output[0]
        } else {
            val output = Array(1) { ByteArray(classes) }
            outputs[0] = output
            member.interpreter.runForMultipleInputsOutputs(inputs, outputs)
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
                    expectedOutputClasses = classes,
                    useZ = asset.useZ
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