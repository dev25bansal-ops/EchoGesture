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
    fun allFourMembersAllocateWithTheFrozenTensorContract() {
        val names = ModelContract.MEMBER_SHA256.keys.toList()
        assertEquals("four pinned members", 4, names.size)

        val interpreters = names.map { interpreterFor(readAsset(it)) }
        try {
            assertEquals("one interpreter per member", 4, interpreters.size)
            interpreters.forEachIndexed { i, interpreter ->
                assertEquals("${names[i]} input tensors", 1, interpreter.inputTensorCount)
                assertEquals("${names[i]} output tensors", 1, interpreter.outputTensorCount)
                assertArrayEquals(
                    "${names[i]} input shape",
                    intArrayOf(1, ModelContract.EXPECTED_FRAMES, ModelContract.EXPECTED_FEATURE_DIM),
                    interpreter.getInputTensor(0).shape()
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
    @Test
    fun productionClassifierReproducesPythonOnFrozenFeatures() {
        val fixture = loadFixture()
        val loaded = ModelContract.load(instrumentation.targetContext.assets)
        assertEquals("contract must resolve all four members", 4, loaded.members.size)
        assertEquals(
            "member order must match the fixture metadata",
            fixture.members,
            loaded.members.map { it.name }
        )

        val classifier = LandmarkClassifier.create(loaded)
        try {
            // Feed the fixture's real feature vector through every member.
            // A zero tensor would run the interpreters but would not exercise
            // the trained input, and would land on some unrelated class.
            val memberLogits = classifier.memberLogits(fixture.features)
            assertEquals("per-member logits collected", 4, memberLogits.size)

            // Production aggregation on real per-member outputs must land on the
            // reference answer. This is aggregation over REAL inference, not a
            // frozen-logit fixture.
            val averaged = EnsembleAggregation.meanLogits(memberLogits.toTypedArray())
            val probs = EnsembleAggregation.softmax(averaged)
            val top = EnsembleAggregation.topK(probs, 5)
            val labels = top.map { loaded.labels[it] }
            assertEquals("top-1 gloss", fixture.top5[0], labels[0])
            assertEquals("top-5 glosses", fixture.top5, labels)
            assertEquals("top-1 confidence", fixture.confidence, probs[top[0]], 1e-4f)
            assertTrue(
                "classifier must report all four members",
                classifier.memberCount == 4
            )
        } finally {
            classifier.close()
        }
    }

    /** Runs one member on a deterministic zero input; returns its logits. */
    private fun averageLogitsMainShape(bytes: ByteArray): FloatArray {
        val interpreter = interpreterFor(bytes)
        return try {
            val shape = interpreter.getInputTensor(0).shape()
            require(shape.contentEquals(
                intArrayOf(1, ModelContract.EXPECTED_FRAMES, ModelContract.EXPECTED_FEATURE_DIM)
            )) { "unexpected input shape ${shape.toList()}" }
            val input = Array(1) {
                Array(ModelContract.EXPECTED_FRAMES) {
                    FloatArray(ModelContract.EXPECTED_FEATURE_DIM)
                }
            }
            val output = Array(1) { FloatArray(ModelContract.EXPECTED_CLASSES) }
            interpreter.run(input, output)
            output[0]
        } finally {
            interpreter.close()
        }
    }
}