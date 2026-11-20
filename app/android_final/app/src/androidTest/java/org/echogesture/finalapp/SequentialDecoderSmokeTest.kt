package org.echogesture.finalapp

import android.net.Uri
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Does [SequentialVideoDecoder] survive on this device?
 *
 * This exists BEFORE the decoder is wired into the app, and that order is the
 * whole point. The first attempt at sequential decoding killed the process in
 * `ImageReader$SurfaceImage.nativeCreatePlanes` — a native abort with no Java
 * stack, so nothing could catch it and the app simply vanished mid-analysis.
 * A unit test cannot catch that either, but running it here means the abort
 * happens in the test process before any user's session can hit it.
 *
 * The test asserts two things that matter independently:
 *  1. it returns frames at all (not a silent null), and
 *  2. the frames carry real image content, not a blank bitmap.
 *
 * Content is checked with a cheap proxy: a decoded camera/ASL frame has a wide
 * spread of luma values. An all-black or all-one-colour buffer — which is what a
 * mis-scaled YUV conversion produces — has almost none.
 */
class SequentialDecoderSmokeTest {

    /** A detected frame sequence, with how many frames carried a hand. */
    class Detected(val frames: List<HandTracker.Frame>, val handFrames: Int)

    companion object {

    /**
     * Runs the shipped detector over already-decoded bitmaps.
     *
     * Shared so the smoke test and the record-mode test cannot disagree about
     * how a frame becomes landmarks: same options, same handedness mapping, same
     * strictly-increasing timestamps.
     */
    /**
     * A detector held open across calls.
     *
     * MEASURED on the Vivo V2404 (2026-11-20): MediaPipe 0.10.14 cannot create a
     * second HandLandmarker after the first is closed, in the same process --
     * `round 2: creating` never returns, with mediapipe_gl_ru alive and the
     * process idle at 0% CPU. DetectorReuseProbeTest proves it.
     *
     * That is a HARNESS limitation, not a product defect: the shipped app builds
     * one landmarker per HandTracker and keeps it for the process's life, which
     * is why the camera works. Only test code hits the recreate path.
     *
     * So tests that classify several clips must create ONE detector and reuse
     * it. This wrapper exists to make that the easy thing to do.
     */
    class Detector private constructor(
        private val lm: com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
    ) : AutoCloseable {
        private var ts = 0L

        fun detect(bitmaps: List<android.graphics.Bitmap>): Detected {
            val out = ArrayList<HandTracker.Frame>(bitmaps.size)
            var hands = 0
            for ((i, bmp) in bitmaps.withIndex()) {
                val img = com.google.mediapipe.framework.image.BitmapImageBuilder(bmp).build()
                val r = lm.detectForVideo(img, ts)
                ts += 33
                img.close()
                val hs = ArrayList<HandTracker.Hand>(r.landmarks().size)
                for (k in r.landmarks().indices) {
                    val cat = if (k < r.handedness().size && r.handedness()[k].isNotEmpty())
                        r.handedness()[k][0].categoryName() else "Right"
                    val slot = if (cat.equals("Left", ignoreCase = true)) 0 else 1
                    val arr = FloatArray(21 * 3)
                    for (j in 0 until 21) {
                        arr[j * 3] = r.landmarks()[k][j].x()
                        arr[j * 3 + 1] = r.landmarks()[k][j].y()
                        arr[j * 3 + 2] = r.landmarks()[k][j].z()
                    }
                    hs.add(HandTracker.Hand(slot, arr))
                }
                if (hs.isNotEmpty()) hands++
                // Absolute time, monotonically increasing across calls.
                out.add(HandTracker.Frame(hs, i * 33L))
            }
            return Detected(out, hands)
        }

        override fun close() = lm.close()

        companion object {
            fun open(ctx: android.content.Context): Detector {
                val opts = com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
                    .HandLandmarkerOptions.builder()
                    .setBaseOptions(
                        com.google.mediapipe.tasks.core.BaseOptions.builder()
                            .setModelAssetPath(ModelContract.DETECTOR_ASSET)
                            .setDelegate(com.google.mediapipe.tasks.core.Delegate.CPU)
                            .build()
                    )
                    .setRunningMode(com.google.mediapipe.tasks.vision.core.RunningMode.VIDEO)
                    .setNumHands(2)
                    .setMinHandDetectionConfidence(0.3f)
                    .setMinHandPresenceConfidence(0.3f)
                    .setMinTrackingConfidence(0.3f)
                    .build()
                return Detector(
                    com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
                        .createFromOptions(ctx, opts)
                )
            }
        }
    }

    fun detectFrames(
        ctx: android.content.Context,
        bitmaps: List<android.graphics.Bitmap>
    ): Detected {
        val opts = com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
            .HandLandmarkerOptions.builder()
            .setBaseOptions(
                com.google.mediapipe.tasks.core.BaseOptions.builder()
                    .setModelAssetPath(ModelContract.DETECTOR_ASSET)
                    // CPU: GPU + IMAGE in an instrumentation process is the
                    // combination that hangs or aborts on this device.
                    .setDelegate(com.google.mediapipe.tasks.core.Delegate.CPU)
                    .build()
            )
            .setRunningMode(com.google.mediapipe.tasks.vision.core.RunningMode.VIDEO)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.3f)
            .setMinHandPresenceConfidence(0.3f)
            .setMinTrackingConfidence(0.3f)
            .build()
        val lm = com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
            .createFromOptions(ctx, opts)
        val out = ArrayList<HandTracker.Frame>(bitmaps.size)
        var hands = 0
        var ts = 0L
        try {
            for ((i, bmp) in bitmaps.withIndex()) {
                val img = com.google.mediapipe.framework.image.BitmapImageBuilder(bmp).build()
                val r = lm.detectForVideo(img, ts)
                ts += 33
                img.close()
                val hs = ArrayList<HandTracker.Hand>(r.landmarks().size)
                for (k in r.landmarks().indices) {
                    val cat = if (k < r.handedness().size && r.handedness()[k].isNotEmpty())
                        r.handedness()[k][0].categoryName() else "Right"
                    val slot = if (cat.equals("Left", ignoreCase = true)) 0 else 1
                    val arr = FloatArray(21 * 3)
                    for (j in 0 until 21) {
                        arr[j * 3] = r.landmarks()[k][j].x()
                        arr[j * 3 + 1] = r.landmarks()[k][j].y()
                        arr[j * 3 + 2] = r.landmarks()[k][j].z()
                    }
                    hs.add(HandTracker.Hand(slot, arr))
                }
                if (hs.isNotEmpty()) hands++
                // 33 ms per frame: a nominal 30 fps timeline, which is what the
                // recorder and segmenter both assume.
                out.add(HandTracker.Frame(hs, i * 33L))
            }
        } finally {
            lm.close()
        }
        return Detected(out, hands)
    }
    }

    private fun report(line: String) {
        println(line)
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        runCatching { File(ctx.filesDir, "seq_decoder_smoke.txt").appendText(line + "\n") }
    }

    @Test
    fun decodesARealClipWithoutCrashing() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        runCatching { File(ctx.filesDir, "seq_decoder_smoke.txt").writeText("") }

        val dirs = listOfNotNull(ctx.getExternalFilesDir(null), ctx.filesDir)
        val clip = dirs
            .flatMap { d ->
                d.listFiles { f -> f.isFile && f.name.endsWith(".mp4", true) }?.toList()
                    ?: emptyList()
            }
            .sortedBy { it.name }
            .firstOrNull()
        assumeTrue("no clip on device; run is INCOMPLETE", clip != null)

        report("clip=${clip!!.name} size=${clip.length()}")
        val t0 = android.os.SystemClock.elapsedRealtime()
        val result = SequentialVideoDecoder.decode(
            ctx, Uri.fromFile(clip), targetFps = 30, maxEdge = 640, maxFrames = 120
        )
        val wall = android.os.SystemClock.elapsedRealtime() - t0
        report("returned=${result != null} wall=${wall}ms")
        assertTrue("decoder returned null (codec refused the file)", result != null)

        val r = result!!
        report("frames=${r.bitmaps.size} pts=${r.ptsMs.size} codec=${r.codecName} " +
            "decodeMs=${r.decodeMs}")
        report("diag ${r.diag}")
        assertTrue("decoder produced no frames", r.bitmaps.isNotEmpty())
        assertTrue(
            "bitmaps and pts lengths disagree: ${r.bitmaps.size} vs ${r.ptsMs.size}",
            r.bitmaps.size == r.ptsMs.size
        )

        // Content check: luma spread across a sample of pixels.
        val bmp = r.bitmaps[r.bitmaps.size / 2]
        val w = bmp.width
        val h = bmp.height
        report("sample frame ${w}x${h}")
        var min = 255
        var max = 0
        var sum = 0L
        var n = 0
        for (yy in 0 until h step 7) {
            for (xx in 0 until w step 7) {
                val p = bmp.getPixel(xx, yy)
                val luma = ((p shr 16 and 0xFF) * 299 +
                    (p shr 8 and 0xFF) * 587 + (p and 0xFF) * 114) / 1000
                if (luma < min) min = luma
                if (luma > max) max = luma
                sum += luma
                n++
            }
        }
        val mean = if (n > 0) sum / n else 0
        report("luma min=$min max=$max mean=$mean spread=${max - min}")
        assertTrue(
            "frame looks blank (luma spread ${max - min}); the YUV conversion is probably wrong",
            (max - min) > 30
        )

        for (b in r.bitmaps) runCatching { b.recycle() }
        report("OK")
    }
}