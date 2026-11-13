package org.echogesture.finalapp

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM parity test for ensemble AGGREGATION only.
 *
 * Fixture `parity_ensemble_logits.txt` was produced by the Python reference
 * (src/main.py::_run_ensemble) on the real clip
 * evaluation/device/v5_accuracy/data/val_30/0283886564670357-WHAT FOR.mp4:
 *
 *   line 0: "<members> <classes>"
 *   line 1: member basenames in meta order
 *   lines 2..5: raw float32 logits per member (real TFLite output)
 *   line 6: averaged logits from numpy mean
 *   line 7: softmax probabilities
 *   line 8: expected top-5 indices
 *   line 9: expected top-5 labels
 *   line 10: expected top-1 confidence
 *   line 11: full 100-entry vocabulary (label index -> label), from
 *            models/landmark_labels.json
 *
 * This exercises [EnsembleAggregation] (mean -> one softmax -> top-5). It does
 * NOT run TFLite: JVM unit tests have no interpreter and the model binaries are
 * not on the unit-test classpath, so per-member tensor-index resolution and
 * on-device inference remain unverified here.
 */
class EnsembleParityTest {

    private fun loadFixture(name: String): List<String> =
        javaClass.classLoader!!.getResource("$name.txt")
            ?.readText()
            ?.lineSequence()
            ?.filter { it.isNotBlank() }
            ?.toList()
            ?: error("missing fixture $name.txt")

    private fun parseFloats(line: String, expectedCount: Int): FloatArray {
        val values = line.trim().split(' ').filter { it.isNotEmpty() }
        assertEquals("float count", expectedCount, values.size)
        return FloatArray(values.size) { values[it].toFloat() }
    }

    @Test
    fun aggregationMatchesPythonReference() {
        val lines = loadFixture("parity_ensemble_logits")
        val (members, classes) = lines[0].split(' ').let { it[0].toInt() to it[1].toInt() }
        assertEquals("fixture member count", 4, members)
        assertEquals("fixture class count", 100, classes)
        assertEquals(
            "member order must match landmark_model_meta.json",
            listOf(
                "landmark_ens_tcn21_fp16.tflite",
                "landmark_ens_tcnm61_fp16.tflite",
                "landmark_ens_tcnm62_fp16.tflite",
                "landmark_ens_rob71_fp16.tflite"
            ),
            lines[1].trim().split(' ')
        )

        val memberLogits = Array(members) { i -> parseFloats(lines[2 + i], classes) }
        val expectedAvg = parseFloats(lines[2 + members], classes)
        val expectedProbs = parseFloats(lines[3 + members], classes)
        val expectedTop = lines[4 + members].trim().split(' ').map { it.toInt() }
        val expectedLabels = lines[5 + members].trim().split(' ')
        val expectedConfidence = lines[6 + members].trim().toFloat()
        val vocabulary = lines[7 + members].trim().split(' ')
        assertEquals("vocabulary size", classes, vocabulary.size)

        // 1) logits are averaged before softmax, element-wise.
        val avg = EnsembleAggregation.meanLogits(memberLogits)
        assertArrayEquals(
            "averaged logits drift from numpy mean(member_logits)",
            expectedAvg, avg, 1e-5f
        )

        // 2) exactly one softmax over the averaged logits.
        val probs = EnsembleAggregation.softmax(avg)
        assertArrayEquals(
            "softmax over averaged logits drifts from the Python path",
            expectedProbs, probs, 1e-6f
        )

        // 3) top-5 indices and label mapping through the same vocabulary the
        // app loads from landmark_labels.json.
        val top = EnsembleAggregation.topK(probs, 5)
        assertArrayEquals("top-5 indices drift", expectedTop.toIntArray(), top)
        val mappedTop5 = top.map { vocabulary[it] }
        assertEquals("top-5 labels must map from indices", expectedLabels, mappedTop5)
        assertEquals("top-1 label", "WHATFOR1", mappedTop5[0])

        // 4) confidence is the softmax probability of the top-1 class.
        assertEquals(
            "top-1 confidence drift", expectedConfidence, probs[top[0]], 1e-6f
        )
        assertEquals("confidence is 0.8821 on the reference clip", 0.8821f, probs[top[0]], 5e-5f)

        // 5) The margin the app gates acceptance on is top-1 minus top-2 over
        // the same softmax. On this clip the runner-up scores 0.0173, so the
        // margin is 0.8648 — far above the 0.15 cut, i.e. this reference clip
        // is accepted. Asserting the value keeps the two quantities the UI
        // depends on tied to the real Python reference.
        val margin = probs[top[0]] - probs[top[1]]
        assertEquals("margin is top1-top2 on the reference clip", 0.8648f, margin, 1e-3f)
        assertTrue("the reference clip must clear the acceptance margin", margin >= 0.15f)
    }

    @Test
    fun averagingIsOrderIndependentButMemberCountSensitive() {
        val lines = loadFixture("parity_ensemble_logits")
        val classes = lines[0].split(' ')[1].toInt()
        val memberLogits = Array(4) { i -> parseFloats(lines[2 + i], classes) }

        // Order must not change the mean (Python uses np.mean over axis 0).
        // Float addition is not associative, so allow a last-bit difference.
        val reordered = arrayOf(memberLogits[2], memberLogits[0], memberLogits[3], memberLogits[1])
        assertArrayEquals(
            "mean must be independent of member order",
            EnsembleAggregation.meanLogits(memberLogits),
            EnsembleAggregation.meanLogits(reordered),
            1e-6f
        )

        // Dropping a member changes the mean: an incomplete asset set is a real
        // behavioral change, not a cosmetic one.
        val threeMember = EnsembleAggregation.meanLogits(memberLogits.take(3).toTypedArray())
        val fourMember = EnsembleAggregation.meanLogits(memberLogits)
        var differs = 0
        for (i in 0 until classes) if (threeMember[i] != fourMember[i]) differs++
        assertTrue("dropping a member must change averaged logits", differs > 50)
    }

    @Test
    fun rejectsMismatchedMemberShapes() {
        val ok = FloatArray(100) { it.toFloat() }
        val bad = FloatArray(99) { it.toFloat() }
        try {
            EnsembleAggregation.meanLogits(arrayOf(ok, bad))
            error("expected IllegalArgumentException for mismatched logits")
        } catch (expected: IllegalArgumentException) {
            assertTrue(expected.message!!.contains("expected 100"))
        }
    }

    @Test
    fun softmaxIsStableForLargeLogits() {
        val logits = FloatArray(100) { 1000f }
        val probs = EnsembleAggregation.softmax(logits)
        var sum = 0f
        probs.forEach { sum += it }
        assertEquals("softmax must normalize", 1f, sum, 1e-5f)
        assertEquals("uniform logits -> uniform probs", 0.01f, probs[0], 1e-6f)
    }
}