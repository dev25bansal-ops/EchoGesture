package org.echogesture.finalapp

import android.graphics.Bitmap
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * Does enabling the GPU delegate change what the app predicts?
 *
 * This is the one question the GPU change left open. The 85.45% headline was
 * measured in PYTHON with MediaPipe's CPU path on desktop; the phone now runs
 * the GPU delegate, and a delegate can produce different landmarks, which would
 * change every downstream feature. Until this runs, GPU landmark parity is
 * unmeasured — not disproven, just unchecked.
 *
 * Method: decode each clip ONCE, then run the SAME frames through two
 * HandLandmarkers (GPU and CPU), build features with the shipped
 * FeatureTransform, and compare (a) the predicted gloss, (b) the top-1
 * probability, and (c) how many frames saw a hand. Matching answers mean the
 * delegate is free; a divergence would mean the shipped accuracy figure does
 * not transfer to the phone.
 *
 * Kept deliberately separate from ClipReplayAccuracyTest (which hangs on this
 * device) and small — 6 clips, one decode each — so it finishes well inside an
 * instrumentation timeout.
 */
class GpuCpuParityTest {

    private val maxClips = 6

    @Test
    fun gpuAndCpuDelegatesAgreeOnTheSameFrames() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = listOfNotNull(ctx.getExternalFilesDir(null), ctx.filesDir).firstOrNull { d ->
            d.listFiles { f -> f.isFile && f.name.endsWith(".mp4", true) }?.isNotEmpty() == true
        }
        assumeTrue("no clip corpus pushed; this run is INCOMPLETE", dir != null)
        val files = dir!!.listFiles { f -> f.isFile && f.name.endsWith(".mp4", true) }!!
            .sortedBy { it.name }.take(maxClips)
        assumeTrue("clip corpus empty", files.isNotEmpty())

        val loaded = ModelContract.load(ctx.assets)
        val classifier = LandmarkClassifier.create(loaded)

        var agree = 0
        var compared = 0
        var maxProbDelta = 0.0

        try {
            for (f in files) {
                val dec = VideoFrames.sample(f, 24)
                if (dec.frames.isEmpty()) {
                    println("PARITY ${f.name} SKIP no_frames completed=${dec.completed}")
                    continue
                }
                if (!dec.completed) {
                    println("PARITY ${f.name} WARN decode_incomplete " +
                        "frames=${dec.frames.size} decoded=${dec.decodedCount}")
                }
                val gpu = runDelegate(dec.frames, Delegate.GPU, classifier)
                val cpu = runDelegate(dec.frames, Delegate.CPU, classifier)
                for (d in dec.frames) runCatching { d.bitmap.recycle() }
                if (gpu == null || cpu == null) {
                    println("PARITY ${f.name} SKIP delegate_unavailable")
                    continue
                }
                compared++
                val same = gpu.first == cpu.first
                if (same) agree++
                val delta = abs(gpu.second - cpu.second).toDouble()
                if (delta > maxProbDelta) maxProbDelta = delta
                println(
                    "PARITY ${f.name} gpu=${gpu.first}(${"%.4f".format(gpu.second)},h=${gpu.third}) " +
                        "cpu=${cpu.first}(${"%.4f".format(cpu.second)},h=${cpu.third}) " +
                        (if (same) "AGREE" else "DIFFER")
                )
            }
        } finally {
            classifier.close()
        }

        println(
            "PARITY SUMMARY agree=$agree compared=$compared " +
                "max_prob_delta=${"%.4f".format(maxProbDelta)} clips=${files.size}"
        )
        assertTrue("no clip could be compared (delegates or corpus unavailable)", compared > 0)
        // Every prediction must match. A delegate that changes answers changes
        // the shipped accuracy, which would have to be re-measured before any
        // figure in the deck could be trusted.
        assertEquals(
            "GPU and CPU delegates must predict the same gloss for the same frames; " +
                "see the PARITY lines above",
            compared, agree
        )
    }

    /** Returns (prediction, top1Probability, framesWithHands) or null if unavailable. */
    private fun runDelegate(
        frames: List<VideoFrames.Decoded>,
        delegate: Delegate,
        classifier: LandmarkClassifier,
    ): Triple<String, Float, Int>? {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val landmarker = try {
            HandLandmarker.createFromOptions(
                ctx,
                HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(
                        BaseOptions.builder()
                            .setModelAssetPath(ModelContract.DETECTOR_ASSET)
                            .setDelegate(delegate)
                            .build()
                    )
                    .setRunningMode(RunningMode.IMAGE)
                    .setNumHands(2)
                    .setMinHandDetectionConfidence(0.3f)
                    .setMinHandPresenceConfidence(0.3f)
                    .setMinTrackingConfidence(0.3f)
                    .build()
            )
        } catch (t: Throwable) {
            println("PARITY delegate=$delegate unavailable: ${t.javaClass.simpleName}")
            return null
        }
        val n = frames.size
        val landmarks = FloatArray(n * 2 * 21 * 3)
        val present = FloatArray(n * 2)
        var hands = 0
        try {
            for ((fi, df) in frames.withIndex()) {
                val copy = df.bitmap.copy(Bitmap.Config.ARGB_8888, false)
                val img = BitmapImageBuilder(copy).build()
                val result = landmarker.detect(img)
                img.close()
                for (k in result.landmarks().indices) {
                    val cat = result.handedness()[k].firstOrNull()?.categoryName() ?: "Right"
                    val slot = if (cat.equals("Left", ignoreCase = true)) 0 else 1
                    present[fi * 2 + slot] = 1f
                    val lm = result.landmarks()[k]
                    val base = (fi * 2 + slot) * 21 * 3
                    for (j in 0 until 21) {
                        landmarks[base + j * 3] = lm[j].x()
                        landmarks[base + j * 3 + 1] = lm[j].y()
                        landmarks[base + j * 3 + 2] = lm[j].z()
                    }
                }
                if (result.landmarks().isNotEmpty()) hands++
            }
        } finally {
            landmarker.close()
        }
        val xy = FeatureTransform.build(landmarks, present, n, 24, useZ = false)
        val xyz = FeatureTransform.build(landmarks, present, n, 24, useZ = true)
        val r = classifier.predict(xy.features, xy.mask, xyz.features)
        return Triple(r.prediction, r.confidence, hands)
    }
}