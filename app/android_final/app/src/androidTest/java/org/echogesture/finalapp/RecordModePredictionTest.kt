package org.echogesture.finalapp

import android.net.Uri
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Does START SIGN -> perform -> FINISH produce a correct gloss?
 *
 * The existing RecordModeOnDeviceTest covers the recorder's MECHANICS (no frames
 * before Start, no frames after Stop, a camera switch ends it). It never asks
 * whether the thing a user actually does — sign at the camera — gives the right
 * answer.
 *
 * This test drives the same chain the app drives, in the same order:
 *
 *   clip -> landmarks -> SignRecorder.Recording -> RecordingFeatures.buildBoth
 *        -> LandmarkClassifier.predict -> ResultPolicy.evaluate
 *
 * The one thing it cannot reproduce is the camera itself. It substitutes a
 * decoded clip for live frames, which exercises everything AFTER detection —
 * exactly the part that record mode owns. Detection is already covered by
 * SequentialDecoderSmokeTest and GpuCpuParityTest.
 *
 * It also checks the question the user actually asks, "did it finish", by
 * running AutoFinishDetector over the same per-frame displacements the live
 * ticker feeds it.
 */
class RecordModePredictionTest {

    private fun report(line: String) {
        println(line)
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        runCatching { File(ctx.filesDir, "record_mode_prediction.txt").appendText(line + "\n") }
    }

    /** The clips this project measured, with the answer each must produce. */
    private val expected = mapOf(
        "NOON" to "NOON1",
        "VOICE" to "VOICE",
        "DARK" to "DARK1",
        "MECHANIC" to "MECHANIC1",
    )

    @Test
    fun aRecordedSignClassifiesCorrectlyThroughTheShippedPath() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        runCatching { File(ctx.filesDir, "record_mode_prediction.txt").writeText("") }

        val dirs = listOfNotNull(ctx.getExternalFilesDir(null), ctx.filesDir)
        val clips = dirs.flatMap { d ->
            d.listFiles { f -> f.isFile && f.name.endsWith(".mp4", true) }?.toList()
                ?: emptyList()
        }.sortedBy { it.name }
        assumeTrue("no clips on device; run is INCOMPLETE", clips.isNotEmpty())

        val loaded = ModelContract.load(ctx.assets)
        val classifier = LandmarkClassifier.create(loaded)
        val checked = ArrayList<String>()
        try {
            for (clip in clips) {
                val key = expected.keys.firstOrNull { clip.name.contains(it, true) }
                    ?: continue
                val want = expected.getValue(key)

                val dec = SequentialVideoDecoder.decode(
                    ctx, Uri.fromFile(clip), targetFps = 30, maxEdge = 640, maxFrames = 120
                )
                if (dec == null || dec.bitmaps.size < 10) {
                    report("$key SKIP decode gave ${dec?.bitmaps?.size ?: 0} frames")
                    continue
                }

                // Detect on those pixels, exactly as the camera path would.
                val s = SequentialDecoderSmokeTest.detectFrames(ctx, dec.bitmaps)
                dec.bitmaps.forEach { runCatching { it.recycle() } }
                report("$key frames=${s.frames.size} hands=${s.handFrames}")

                if (s.frames.size < SignRecorder.MIN_FRAMES) {
                    report("$key SKIP below MIN_FRAMES (${SignRecorder.MIN_FRAMES})")
                    continue
                }

                // Build the Recording a Start/Finish interval would have held.
                val startMs = s.frames.first().capturedAtMs
                val stopMs = s.frames.last().capturedAtMs
                val recording = SignRecorder.Recording(s.frames, startMs, stopMs)

                // The shipped chain.
                val tensors = RecordingFeatures.buildBoth(recording)
                val r = classifier.predict(tensors.featuresXY, tensors.mask, tensors.featuresXYZ)
                val accepted = ResultPolicy.isAccepted(r.margin, r.confidence * 100f)

                report(
                    "$key -> ${r.prediction} conf=${"%.4f".format(r.confidence)} " +
                        "margin=${"%.4f".format(r.margin)} agree=${r.agreement}/${r.memberCount} " +
                        "accepted=$accepted want=$want"
                )
                assertEquals(
                    "record-mode path predicted ${r.prediction} for the $key clip",
                    want, r.prediction
                )
                checked.add(key)
            }
        } finally {
            classifier.close()
        }

        report("checked=${checked.size} ${checked.joinToString(",")}")
        assertTrue("no clip reached classification; the test proved nothing", checked.isNotEmpty())
    }

    @Test
    fun autoFinishWouldEndTheRecordingAfterTheSign() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dirs = listOfNotNull(ctx.getExternalFilesDir(null), ctx.filesDir)
        val clip = dirs.flatMap { d ->
            d.listFiles { f -> f.isFile && f.name.endsWith(".mp4", true) }?.toList()
                ?: emptyList()
        }.sortedBy { it.name }.firstOrNull()
        assumeTrue("no clip on device", clip != null)

        val dec = SequentialVideoDecoder.decode(
            ctx, Uri.fromFile(clip!!), targetFps = 30, maxEdge = 640, maxFrames = 120
        )
        assumeTrue("decode unavailable", dec != null && dec.bitmaps.size >= 10)
        val s = SequentialDecoderSmokeTest.detectFrames(ctx, dec!!.bitmaps)
        dec.bitmaps.forEach { runCatching { it.recycle() } }
        assumeTrue("too few frames", s.frames.size >= 10)

        // Feed the detector exactly as the live ticker does: one displacement
        // per frame, with a wall clock and a recorded duration.
        val det = AutoFinishDetector()
        val d = SignSegmenter.displacement(s.frames)
        var verdict = AutoFinishDetector.Verdict.CONTINUE
        var now = 0L
        for (i in s.frames.indices) {
            now = s.frames[i].capturedAtMs - s.frames.first().capturedAtMs
            verdict = det.onFrame(d[i], now, now)
            if (verdict != AutoFinishDetector.Verdict.CONTINUE) break
        }
        report("autoFinish clip=${clip.name} verdict=$verdict after ${now}ms")
        // A whole signed clip must reach FINISHED (or TIMED_OUT for a very long
        // one) -- never sit at CONTINUE forever, which is what "the app never
        // finishes on its own" would look like.
        assertTrue(
            "auto-finish never fired on a complete clip; verdict=$verdict",
            verdict != AutoFinishDetector.Verdict.CONTINUE
        )
    }
}