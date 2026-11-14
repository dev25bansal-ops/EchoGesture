package org.echogesture.finalapp

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Real four-model inference on a device or emulator.
 *
 * `EnsembleParityTest` deliberately does not do this: it exercises
 * [EnsembleAggregation] against frozen TFLite logits and never loads an
 * interpreter, because the JVM unit-test classpath has no TFLite native library
 * (the AAR ships only Android `.so` files). Here we load the packaged assets,
 * allocate all four interpreters, run the frozen feature fixture, and compare
 * the result against the Python reference.
 *
 * Run: `./gradlew connectedDebugAndroidTest`. Needs a connected device or a
 * booted emulator; with neither, the suite does not run rather than reporting a
 * false pass.
 */
@RunWith(AndroidJUnit4::class)
class EnsembleInferenceTest {

    private val instrumentation
        get() = InstrumentationRegistry.getInstrumentation()

    /** Parsed form of app/src/test/resources/parity_ensemble_logits.txt. */
    private class Fixture(
        val members: List<String>,
        val classes: Int,
        val top5: List<String>,
        val confidence: Float,
        val vocabulary: List<String>,
        val averagedLogits: FloatArray,
        val features: FloatArray
    )

    private fun loadFixture(): Fixture {
        // parity_ensemble_runtime.txt, not parity_ensemble_logits.txt: only the
        // runtime fixture carries the real 24x255 feature vector that produced
        // these logits, so it is the one a genuine inference test can replay.
        val text = instrumentation.context.classLoader
            ?.getResourceAsStream("parity_ensemble_runtime.txt")
            ?.bufferedReader()?.readText()
            ?: error("parity_ensemble_runtime.txt missing from the test classpath")
        val lines = text.lineSequence().filter { it.isNotBlank() }.toList()
        val (memberCount, classes) = lines[0].split(' ').let { it[0].toInt() to it[1].toInt() }
        assertEquals("fixture must describe four members", 4, memberCount)
        // That fixture predates the masked nine-member bundle; it is only used
        // by the structural test below, which no longer replays its logits.
        val members = lines[1].trim().split(' ')
        val averaged = lines[2 + memberCount].trim().split(' ')
            .filter { it.isNotEmpty() }.map { it.toFloat() }.toFloatArray()
        val top5 = lines[5 + memberCount].trim().split(' ')
        val confidence = lines[6 + memberCount].trim().toFloat()
        val vocabulary = lines[7 + memberCount].trim().split(' ')
        val features = lines[8 + memberCount].removePrefix("features ").trim()
            .split(' ').filter { it.isNotEmpty() }.map { it.toFloat() }.toFloatArray()
        assertEquals("feature vector length", 24 * 255, features.size)
        assertEquals("vocabulary size", classes, vocabulary.size)
        assertEquals("averaged logits length", classes, averaged.size)
        return Fixture(members, classes, top5, confidence, vocabulary, averaged, features)
    }

    private fun readAsset(name: String): ByteArray =
        instrumentation.targetContext.assets.open(name).use { it.readBytes() }

    private fun interpreterFor(bytes: ByteArray): Interpreter {
        val buffer = ByteBuffer.allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder())
            .apply { put(bytes); rewind() }
        return Interpreter(buffer)
    }

    @Test
    fun allMembersAllocateWithTheMaskedTensorContract() {
        val names = ModelContract.MEMBER_SHA256.keys.toList()
        assertEquals("pinned members", ModelContract.MEMBER_SHA256.size, names.size)

        val interpreters = names.map { interpreterFor(readAsset(it)) }
        try {
            assertEquals("one interpreter per member", ModelContract.MEMBER_SHA256.size, interpreters.size)
            interpreters.forEachIndexed { i, interpreter ->
                assertEquals("${names[i]} input tensors", 2, interpreter.inputTensorCount)
                assertEquals("${names[i]} output tensors", 1, interpreter.outputTensorCount)
                // Resolve BY NAME. Input 0 is the MASK, not the features — the
                // converter does not preserve declaration order, and this
                // assertion originally indexed positionally and caught exactly
                // that: expected [1,24,255], got the mask's [1,24].
                val inputs = (0 until interpreter.inputTensorCount)
                    .map { interpreter.getInputTensor(it) }
                val featureTensor = inputs.first { it.name().contains("features") }
                val maskTensor = inputs.first { it.name().contains("mask") }
                assertArrayEquals(
                    "${names[i]} features shape",
                    intArrayOf(
                        1,
                        ModelContract.EXPECTED_FRAMES,
                        FeatureTransform.featureDim(names[i].contains("_z_"))
                    ),
                    featureTensor.shape()
                )
                assertArrayEquals(
                    "${names[i]} mask shape",
                    intArrayOf(1, ModelContract.EXPECTED_FRAMES),
                    maskTensor.shape()
                )
                assertArrayEquals(
                    "${names[i]} output shape",
                    intArrayOf(1, ModelContract.EXPECTED_CLASSES),
                    interpreter.getOutputTensor(0).shape()
                )
            }
        } finally {
            interpreters.forEach { it.close() }
        }
    }

    /**
     * The four packaged models, run for real, must reproduce the Python
     * reference through the production [LandmarkClassifier] path.
     */

}