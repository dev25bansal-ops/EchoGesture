package org.echogesture.finalapp

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * ON-DEVICE proof that the shipped app runs FOUR real TFLite interpreters.
 *
 * This is the only test that can execute this claim: the tensorflow-lite AAR
 * ships Android ABIs only, so the JVM unit-test runtime cannot load
 * `tensorflowlite_jni` (verified: UnsatisfiedLinkError). Run with a device or
 * emulator attached:
 *
 *   ./gradlew :app:connectedDebugAndroidTest \
 *       --tests org.echogesture.finalapp.TfliteEnsembleInferenceTest
 *
 * Requires the four FP16 member binaries in `src/main/assets/` (they are not
 * redistributable and therefore absent from a clean checkout). When they are
 * absent the test SKIPS with an explicit message rather than passing vacuously.
 *
 * The fixture `assets/parity_ensemble_runtime.txt` carries the exact 24x255
 * float32 feature vector (hash pinned) plus the Python reference's per-member
 * logits, mean logits, softmax probabilities, top-5 and confidence for the real
 * clip .../0283886564670357-WHAT FOR.mp4, produced by `src/main.py`'s own
 * `_init_model` / `_run_ensemble` path under ai_edge_litert 2.3.0.
 *
 * Tolerances (why):
 *  - per-member / mean logits: 1e-3. The fixture is float32 output from a
 *    different TensorFlow Lite build/runtime on a different CPU. Kernels are
 *    numerically equivalent but not bit-identical across builds; 1e-3 is ~4
 *    decimal digits better than the model's own logit margin (top1-top2 3.93,
 *    top5-top6 0.023), so drift cannot silently change the ranking.
 *  - softmax probabilities: 1e-4.
 *  - top-1 label and top-5 ordering: EXACT.
 *  - confidence: 1e-3 absolute.
 */
@RunWith(AndroidJUnit4::class)
class TfliteEnsembleInferenceTest {

    private fun context(): Context = InstrumentationRegistry.getInstrumentation().targetContext

    private fun fixture(): List<String> {
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        return assets.open("parity_ensemble_runtime.txt").bufferedReader().readLines()
            .filter { it.isNotBlank() && !it.startsWith("#") }
    }

    private fun floats(line: String, n: Int): FloatArray {
        val v = line.trim().split(' ').filter { it.isNotEmpty() }
        assertEquals("float count", n, v.size)
        return FloatArray(n) { v[it].toFloat() }
    }

    @Test
    fun fourRealInterpretersMatchThePythonReference() {
        val assets = context().assets
        val names = assets.list("")?.toList() ?: emptyList()
        val missing = ModelContract.MEMBER_SHA256.keys.filter { it !in names }
        org.junit.Assume.assumeTrue(
            "members not published ($missing absent); device parity test skipped", missing.isEmpty()
        )

        val l = fixture()
        val members = l[0].split(' ')[0].toInt()
        val classes = l[0].split(' ')[1].toInt()
        assertEquals(4, members)
        val expectedNames = l[1].trim().split(' ')
        val expectedLogits = Array(members) { floats(l[2 + it], classes) }
        val expectedAvg = floats(l[2 + members], classes)
        val expectedProbs = floats(l[3 + members], classes)
        val expectedTop = l[4 + members].trim().split(' ').map { it.toInt() }.toIntArray()
        val expectedLabels = l[5 + members].trim().split(' ')
        val expectedConfidence = l[6 + members].trim().toFloat()
        val features = floats(l[7 + 4 + 1].removePrefix("features "), 24 * 255)

        // 1) The real production loader validates and returns all four members.
        val loaded = ModelContract.load(assets)
        assertEquals(4, loaded.members.size)
        assertEquals(expectedNames, loaded.identity.memberNames)
        assertEquals(ModelContract.ENSEMBLE_SHA256, loaded.identity.combinedSha256)

        // 2) Four real interpreters, one per member; closed together.
        LandmarkClassifier.create(loaded).use { classifier ->
            assertEquals("interpreter count", 4, classifier.memberCount)

            // 3) Per-member logits through each member's OWN output tensor.
            val real = classifier.memberLogits(features)
            assertEquals(4, real.size)
            real.forEachIndexed { i, logits ->
                assertEquals(classes, logits.size)
                assertArrayEquals(
                    "member $i (${expectedNames[i]}) logits drift from the Python runtime",
                    expectedLogits[i], logits, 1e-3f
                )
                // Distinct members must not silently be the same model.
                assertTrue(
                    "member $i produced a degenerate (constant) logit vector",
                    logits.max() - logits.min() > 1f
                )
            }
            // The four members are genuinely different models.
            for (i in 0 until 4) for (j in i + 1 until 4) {
                var d = 0f
                for (c in 0 until classes) d = maxOf(d, kotlin.math.abs(real[i][c] - real[j][c]))
                assertTrue("members $i and $j returned identical logits; not four models", d > 0.5f)
            }

            // 4) The full result the user sees.
            val result = classifier.predict(features)
            assertNotNull(result.averagedLogits)
            assertArrayEquals(
                "averaged logits drift", expectedAvg, result.averagedLogits!!, 1e-3f
            )
            assertEquals("top-1 label", expectedLabels[0], result.prediction)
            assertEquals("top-5 labels and order", expectedLabels, result.top5)
            assertEquals("confidence", expectedConfidence, result.confidence, 1e-3f)
            // Softmax must normalize over all 100 classes.
            val probs = EnsembleAggregation.softmax(result.averagedLogits!!)
            var sum = 0.0
            for (c in 0 until classes) sum += probs[c]
            assertEquals("softmax normalizes", 1.0, sum, 1e-4)
            assertArrayEquals("softmax probs", expectedProbs, probs, 1e-4f)
            assertEquals("top-5 indices", expectedTop.toList(), result.top5.map { loaded.labels.indexOf(it) })
        }
    }

    @Test
    fun attentionToTensorIndexResolutionPerMember() {
        // The output tensor handle differs by architecture in the Python
        // flatbuffer (224 for tcn21/rob71, 191 for tcnm61/tcnm62). The Java API
        // addresses positionally, so the invariant is: each Member's Interpreter
        // is the one that produced its logits. A shared/global interpreter or a
        // cross-member read would make these vectors identical; they are not.
        val assets = context().assets
        val names = assets.list("")?.toList() ?: emptyList()
        val missing = ModelContract.MEMBER_SHA256.keys.filter { it !in names }
        org.junit.Assume.assumeTrue("members not published; skipped", missing.isEmpty())

        val l = fixture()
        val classes = l[0].split(' ')[1].toInt()
        val expectedLogits = Array(4) { floats(l[2 + it], classes) }
        val features = floats(l[7 + 4 + 1].removePrefix("features "), 24 * 255)
        LandmarkClassifier.create(ModelContract.load(assets)).use { classifier ->
            val real = classifier.memberLogits(features)
            for (i in 0 until 4) for (j in i + 1 until 4) {
                var d = 0f
                for (c in 0 until classes) d = maxOf(d, kotlin.math.abs(real[i][c] - real[j][c]))
                assertTrue("members $i/$j identical: interpreter/tensor cross-wired", d > 0.5f)
            }
            // And each member matches ITS OWN row of the fixture, not another's.
            real.forEachIndexed { i, logits ->
                assertArrayEquals(expectedLogits[i], logits, 1e-3f)
            }
        }
    }
}