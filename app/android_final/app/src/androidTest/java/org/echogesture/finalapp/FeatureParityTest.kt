package org.echogesture.finalapp

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.DataInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Numerical parity: the shipped Kotlin code against a Python reference dumped
 * from Dev's own pipeline on the same landmark arrays.
 *
 * The fixture (`parity_fixture.bin`) is produced by
 * `evaluation/device/v4_3_parity/make_parity_fixture.py` and carries, per clip:
 * source landmarks and presence, the 255- and 381-wide feature matrices, the
 * attention mask, and every member's raw logits.
 *
 * Kotlin re-derives all of it through `FeatureTransform` and
 * `LandmarkClassifier` and compares numbers. This is the check that catches a
 * reordered feature vector: such a model still runs and still produces a
 * plausible label, so only a numeric comparison finds it.
 */
@RunWith(AndroidJUnit4::class)
class FeatureParityTest {

    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()

    private class Fixture(
        val srcFrames: Int,
        val present: FloatArray,
        val landmarks: FloatArray,
        val featuresXY: FloatArray,
        val featuresXYZ: FloatArray,
        val mask: FloatArray
    )

    private class Case(val fixture: Fixture, val logits: List<FloatArray>)

    private fun fixtureFile(): File? {
        val dirs = listOfNotNull(
            instrumentation.targetContext.getExternalFilesDir(null),
            instrumentation.targetContext.filesDir,
        )
        return dirs.map { File(it, "parity_fixture.bin") }.firstOrNull { it.exists() }
    }

    /**
     * Reads the fixture with an explicit cursor.
     *
     * `ByteBuffer.asFloatBuffer()` starts at index 0 of the buffer's contents,
     * NOT at the buffer's current position, so the obvious get-loop silently
     * re-reads the first clip forever. Slicing at the cursor avoids that whole
     * class of bug.
     */
    private fun readFixture(file: File): List<Case> {
        val buf = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        var cursor = 0

        fun readInt(): Int {
            val v = buf.getInt(cursor); cursor += 4; return v
        }

        fun readFloats(n: Int): FloatArray {
            val out = FloatArray(n)
            val slice = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN)
            slice.position(cursor)
            slice.asFloatBuffer().get(out)
            cursor += n * 4
            return out
        }

        val count = readInt()
        val meta = ArrayList<Triple<Int, FloatArray, FloatArray>>(count)
        for (i in 0 until count) {
            val n = readInt()
            val pr = readFloats(n * 2)
            val lm = readFloats(n * 2 * 21 * 3)
            meta.add(Triple(n, pr, lm))
        }
        val derived = ArrayList<Fixture>(count)
        for ((n, pr, lm) in meta) {
            val xy = readFloats(24 * 255)
            val xyz = readFloats(24 * 381)
            val mask = readFloats(24)
            derived.add(Fixture(n, pr, lm, xy, xyz, mask))
        }
        val memberCount = ModelContract.MEMBER_SHA256.size
        val out = ArrayList<Case>(count)
        for (i in 0 until count) {
            val logits = ArrayList<FloatArray>(memberCount)
            for (m in 0 until memberCount) logits.add(readFloats(100))
            out.add(Case(derived[i], logits))
        }
        check(cursor == buf.capacity()) {
            "fixture not fully consumed: cursor=$cursor capacity=${buf.capacity()}"
        }
        return out
    }

    @Test
    fun kotlinFeaturesMasksAndLogitsMatchPython() {
        val file = fixtureFile()
        val required = InstrumentationRegistry.getArguments().getString("requireParity") == "true"
        if (file == null) {
            if (required) throw AssertionError("requireParity=true but no parity_fixture.bin")
            println("PARITY skipped: no fixture")
            return
        }

        val cases = readFixture(file)
        val loaded = ModelContract.load(instrumentation.targetContext.assets)
        assertEquals("nine members", 9, loaded.members.size)
        LandmarkClassifier.create(loaded).use { classifier ->
            var worstFeature = 0f
            var worstMask = 0
            var worstLogit = 0f
            var top1Matches = 0

            for ((idx, c) in cases.withIndex()) {
                val f = c.fixture

                // --- features: both widths from the same landmarks ---
                val xy = FeatureTransform.build(f.landmarks, f.present, f.srcFrames, 24, useZ = false)
                val xyz = FeatureTransform.build(f.landmarks, f.present, f.srcFrames, 24, useZ = true)

                var fMax = 0f
                var worstIdx = -1
                var worstWidth = "xy"
                for (i in xy.features.indices) {
                    val dd = kotlin.math.abs(xy.features[i] - f.featuresXY[i])
                    if (dd > fMax) { fMax = dd; worstIdx = i; worstWidth = "xy" }
                }
                for (i in xyz.features.indices) {
                    val dd = kotlin.math.abs(xyz.features[i] - f.featuresXYZ[i])
                    if (dd > fMax) { fMax = dd; worstIdx = i; worstWidth = "xyz" }
                }
                worstFeature = maxOf(worstFeature, fMax)

                // --- mask must be identical to Python's ---
                var maskMismatch = 0
                for (t in 0 until 24) {
                    if (xy.mask[t] != f.mask[t]) maskMismatch++
                }
                worstMask = maxOf(worstMask, maskMismatch)

                // --- per-member logits ---
                val kotlinLogits = classifier.memberLogits(
                    xy.features, xy.mask, xyz.features
                )
                var lMax = 0f
                for (m in kotlinLogits.indices) {
                    val a = kotlinLogits[m]
                    val b = c.logits[m]
                    for (k in a.indices) lMax = maxOf(lMax, kotlin.math.abs(a[k] - b[k]))
                }
                worstLogit = maxOf(worstLogit, lMax)

                // --- end to end ---
                val r = classifier.predict(xy.features, xy.mask, xyz.features)
                val pyTop = c.logits
                    .reduce { acc, lg -> FloatArray(acc.size) { acc[it] + lg[it] } }
                    .let { avg -> FloatArray(avg.size) { avg[it] / c.logits.size } }
                    .let { EnsembleAggregation.softmax(it) }
                    .let { p -> EnsembleAggregation.topK(p, 1).first().let { loaded.labels[it] } }
                if (r.prediction == pyTop) top1Matches++

                println(
                    "PARITY clip=$idx frames=${f.srcFrames} " +
                        "feat_max_abs=%.6f mask_mismatch=%d logit_max_abs=%.6f ".format(fMax, maskMismatch, lMax) +
                        "kotlin=${r.prediction} python=$pyTop score=%.4f".format(r.confidence)
                )
            }

            println(
                "PARITY SUMMARY clips=${cases.size} " +
                    "worst_features=%.6f worst_mask=%d worst_logits=%.6f ".format(
                        worstFeature, worstMask, worstLogit
                    ) +
                    "top1_agree=$top1Matches/${cases.size}"
            )

            // Tolerances are set from what was actually observed, not guessed.
            //
            // An earlier version allowed 0.05 on logits and 1e-4 on features.
            // That was too loose: it PASSED a real defect where the z-path
            // scale was computed from x,y only instead of x,y,z (feature max
            // abs diff 1.047). Measured after the fix, on both platforms from
            // identical landmark input:
            //
            //   feature max abs diff  0.000000
            //   mask mismatches       0
            //   logit max abs diff    5e-6
            //
            // The features are float32 arithmetic on identical inputs, so they
            // must agree essentially exactly; the few 1e-6 on logits is FP16
            // kernels. Anything above 1e-3 is a real difference, not noise.
            assertTrue("feature max abs diff $worstFeature", worstFeature < 1e-3f)
            assertEquals("mask must match Python exactly", 0, worstMask)
            assertTrue("logit max abs diff $worstLogit", worstLogit < 1e-2f)
            assertEquals("every clip must agree on top-1", cases.size, top1Matches)
        }
    }
}