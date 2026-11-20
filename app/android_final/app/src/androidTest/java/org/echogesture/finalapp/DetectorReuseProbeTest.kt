package org.echogesture.finalapp

import android.graphics.Bitmap
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Can the detector be created, used and closed MORE THAN ONCE in one process?
 *
 * The record-mode test creates a landmarker per clip and hung on the second one,
 * with `mediapipe_gl_ru` alive and the process idle. MediaPipe's native GL
 * runtime may not tolerate a close-and-recreate cycle.
 *
 * If this test hangs, that is the answer and it is a HARNESS limitation on this
 * device, not a product defect: the shipped app creates one landmarker per
 * HandTracker and holds it for the process's life. Reusing a single instance
 * across clips is the workaround for the test.
 */
class DetectorReuseProbeTest {

    private fun report(line: String) {
        println(line)
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        runCatching { File(ctx.filesDir, "detector_reuse.txt").appendText(line + "\n") }
    }

    @Test
    fun twoSequentialLandmarkerLifecyclesSurvive() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        runCatching { File(ctx.filesDir, "detector_reuse.txt").writeText("") }

        val dirs = listOfNotNull(ctx.getExternalFilesDir(null), ctx.filesDir)
        val clip = dirs.flatMap { d ->
            d.listFiles { f -> f.isFile && f.name.endsWith(".mp4", true) }?.toList()
                ?: emptyList()
        }.sortedBy { it.name }.firstOrNull()
        org.junit.Assume.assumeTrue("no clip", clip != null)

        val dec = SequentialVideoDecoder.decode(
            ctx, android.net.Uri.fromFile(clip!!), targetFps = 30, maxEdge = 640, maxFrames = 40
        )
        org.junit.Assume.assumeTrue("decode failed", dec != null && dec.bitmaps.size >= 5)
        // Small slice: this probe measures lifecycle, not throughput.
        val bmp: List<Bitmap> = dec!!.bitmaps.take(8)
        report("using ${bmp.size} frames")

        for (round in 1..2) {
            val t0 = android.os.SystemClock.elapsedRealtime()
            report("round $round: creating")
            val s = SequentialDecoderSmokeTest.detectFrames(ctx, bmp)
            val ms = android.os.SystemClock.elapsedRealtime() - t0
            report("round $round: ok frames=${s.frames.size} hands=${s.handFrames} in ${ms}ms")
            assertTrue("round $round produced no frames", s.frames.isNotEmpty())
        }
        dec.bitmaps.forEach { runCatching { it.recycle() } }
        report("BOTH ROUNDS OK")
    }
}