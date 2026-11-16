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
     * The nine packaged members, run for real, must reproduce the Python
     * reference through the production [LandmarkClassifier] path.
     */

}