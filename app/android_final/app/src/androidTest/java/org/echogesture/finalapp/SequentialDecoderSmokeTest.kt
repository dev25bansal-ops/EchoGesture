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