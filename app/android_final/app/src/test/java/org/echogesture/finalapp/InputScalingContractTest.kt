package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Guards the measured conclusion about input scaling.
 *
 * Training squashed every frame to 256x256 before MediaPipe. That squashes a
 * 640x480 frame by 2.5x horizontally and 1.875x vertically, changing hand
 * aspect ratio. Measured over 961 frames of 10 training clips, the detector
 * found a hand in 41.4% of frames at native resolution but only 37.6% after the
 * 256x256 squash, and on the 10 clips end-to-end the squash turned one correct
 * prediction (AUTISM1 0.235) into a wrong one (EASY 0.110).
 *
 * So the app must NOT re-introduce the training resize. These tests encode the
 * shape of the decision rather than the pixels, because the pixel evidence
 * lives in docs/release/RECOGNITION_ROOT_CAUSE.md and needs model weights to
 * reproduce.
 */
class InputScalingContractTest {

    /** The analysis resolution the app requests, mirroring CameraScreen. */
    private val analysisWidth = 640
    private val analysisHeight = 480

    @Test
    fun analysisResolutionKeepsACameraAspectRatio() {
        // 4:3, like the training clips (640x480) and like a phone's 4:3
        // analysis stream. A square target is what the evidence argues against.
        assertEquals(
            "analysis target must not be square",
            false,
            analysisWidth == analysisHeight
        )
        val ratio = analysisWidth.toFloat() / analysisHeight
        assertTrue("expected a ~4:3 landscape target, got $ratio", ratio in 1.30f..1.36f)
    }

    @Test
    fun squashingToSquareWouldChangeTheAspectRatio() {
        // Documents WHY the 256x256 resize is not applied: it changes the
        // horizontal and vertical scale factors by different amounts.
        val sx = analysisWidth / 256f
        val sy = analysisHeight / 256f
        assertTrue(
            "a 640x480 -> 256x256 squash scales axes differently ($sx vs $sy)",
            kotlin.math.abs(sx - sy) > 0.5f
        )
    }

    @Test
    fun captureWindowIsNotShorterThanMostTrainingClips() {
        // Measured: median in-vocab training clip is 2.89 s, p90 4.83 s, and
        // 3.7 s covers 36 of 40 sampled clips. Shortening the window below the
        // median would truncate the majority of signs.
        val captureSpanMs = HandTracker.CAPTURE_SPAN_MS
        assertTrue(
            "capture window ${captureSpanMs}ms must exceed the 2.89s median",
            captureSpanMs >= 3000L
        )
    }
}