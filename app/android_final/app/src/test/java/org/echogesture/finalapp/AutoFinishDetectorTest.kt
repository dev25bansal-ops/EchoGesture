package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Auto-finish must end a recording when the sign is over, and must NOT end it
 * while the signer is still signing or merely pausing.
 *
 * These are the two ways this feature can be wrong, and both are worse than the
 * tap it replaces: finishing early truncates a sign, finishing never means the
 * button still has to be pressed. Driving the detector with synthetic motion
 * makes both provable without a phone in hand.
 */
class AutoFinishDetectorTest {

    private val quiet = AutoFinishDetector.QUIET_MS_DEFAULT
    private val minRec = AutoFinishDetector.MIN_RECORDING_MS_DEFAULT

    /** Feeds a motion profile at 100 ms per frame, as the real ticker does. */
    private fun drive(
        d: AutoFinishDetector,
        profile: List<Float>,
        stepMs: Long = 100L
    ): AutoFinishDetector.Verdict {
        var verdict = AutoFinishDetector.Verdict.CONTINUE
        var now = 0L
        for (v in profile) {
            now += stepMs
            verdict = d.onFrame(v, now, now)
            if (verdict != AutoFinishDetector.Verdict.CONTINUE) return verdict
        }
        return verdict
    }

    @Test
    fun finishesAfterMotionStartsThenStops() {
        val d = AutoFinishDetector()
        // 1 s of busy motion, then well over the quiet window of stillness.
        val profile = List(10) { 0.20f } + List(20) { 0.0f }
        assertEquals(
            AutoFinishDetector.Verdict.FINISHED,
            drive(d, profile)
        )
    }

    @Test
    fun doesNotFinishWhileTheSignKeepsMoving() {
        val d = AutoFinishDetector()
        val profile = List(60) { 0.20f }   // 6 s of continuous motion
        assertEquals(
            AutoFinishDetector.Verdict.CONTINUE,
            drive(d, profile)
        )
        assertTrue("motion should be recorded as started", d.hasStartedMotion)
    }

    @Test
    fun doesNotFinishBeforeTheSignStarts() {
        val d = AutoFinishDetector()
        // Standing still getting ready: no motion at all for 4 s.
        val profile = List(40) { 0.0f }
        assertEquals(
            AutoFinishDetector.Verdict.CONTINUE,
            drive(d, profile)
        )
        assertFalse("no motion has happened yet", d.hasStartedMotion)
    }

    @Test
    fun aPauseBeforeTheMinimumDoesNotEndTheRecording() {
        val d = AutoFinishDetector()
        // Motion starts, then a pause that would otherwise satisfy the quiet
        // window, but the whole thing is still under the minimum recording.
        val profile = listOf(0.20f, 0.20f) + List(8) { 0.0f }
        assertEquals(
            AutoFinishDetector.Verdict.CONTINUE,
            drive(d, profile)
        )
    }

    @Test
    fun aBriefHoldInsideASignDoesNotEndIt() {
        val d = AutoFinishDetector()
        // Sign, hold for less than the quiet window, then keep signing.
        val profile = List(8) { 0.20f } +
            List((quiet / 100).toInt() - 2) { 0.0f } +
            List(10) { 0.20f }
        assertEquals(
            AutoFinishDetector.Verdict.CONTINUE,
            drive(d, profile)
        )
    }

    @Test
    fun reportsTimeoutWhenMotionNeverSettles() {
        val d = AutoFinishDetector()
        val profile = List(120) { 0.20f }  // 12 s, past the 8 s ceiling
        assertEquals(
            AutoFinishDetector.Verdict.TIMED_OUT,
            drive(d, profile)
        )
    }

    @Test
    fun aQuietSignStillStarts() {
        // Relative thresholds: a signer with small movements must still be
        // detected. The absolute values are deliberately tiny here.
        val d = AutoFinishDetector()
        val profile = List(10) { 0.002f } + List(20) { 0.0f }
        assertEquals(
            AutoFinishDetector.Verdict.FINISHED,
            drive(d, profile)
        )
    }

    @Test
    fun aLoudSignerDoesNotRaiseTheBarForQuietMotion() {
        // Peak is a running max, so the FIRST big frame sets the scale. A
        // smaller but genuine second phase must still count as motion.
        val d = AutoFinishDetector()
        val profile = List(6) { 0.50f } +
            List(6) { 0.20f } +
            List(20) { 0.0f }
        assertEquals(
            AutoFinishDetector.Verdict.FINISHED,
            drive(d, profile)
        )
    }
}
