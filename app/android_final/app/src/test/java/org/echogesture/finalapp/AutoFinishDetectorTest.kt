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
        // A signer who moves less than average must still be detected. The
        // values here are small but above the sensor-noise floor: 0.002, which
        // an earlier version of this test used, is BELOW NOISE_FLOOR and is
        // correctly rejected as jitter now. Corpus p25 of real per-frame motion
        // is 0.15, so 0.03 is already a very quiet sign.
        val d = AutoFinishDetector()
        val profile = List(10) { 0.03f } + List(20) { 0.0f }
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

    @Test
    fun aSingleGlitchFrameDoesNotRaiseTheQuietThreshold() {
        // THE BUG THIS REPLACES: peak was a raw running max, so one spike frame
        // (a handedness flip makes SignSegmenter.displacement pair the wrong
        // hands, ~0.28 while the sign moves at ~0.03) permanently set the quiet
        // threshold to 0.042 and every real sign frame counted as quiet. The
        // sign was finished about 1 s in, half-performed.
        val d = AutoFinishDetector()
        val profile = List(6) { 0.30f } +           // genuine sign motion
            listOf(0.90f) +                          // one glitch spike
            List(12) { 0.30f } +                     // sign continues
            List(20) { 0.0f }                        // then actually stops
        val v = drive(d, profile)
        assertEquals(
            "a single spike must not make the ongoing sign look quiet",
            AutoFinishDetector.Verdict.FINISHED, v
        )
    }

    @Test
    fun raisingTheHandAndPausingToThinkDoesNotFinish() {
        // THE BUG THIS REPLACES: the start gate could never reject (peak was
        // updated from the current frame before being tested against), so the
        // hand's arrival started the sign; a pause to recall the sign then
        // looked like the end of one.
        val d = AutoFinishDetector()
        // 0.5 s of hand-raise motion, then a long still pause BEFORE any sign.
        val profile = List(5) { 0.10f } + List(40) { 0.0f }
        val v = drive(d, profile)
        assertEquals(
            "a ready-position pause is not a finished sign",
            AutoFinishDetector.Verdict.CONTINUE, v
        )
    }

    @Test
    fun noiseAloneNeverStartsASign() {
        // Sub-floor jitter must not satisfy the start gate.
        val d = AutoFinishDetector()
        val profile = List(40) { 0.002f }
        assertEquals(AutoFinishDetector.Verdict.CONTINUE, drive(d, profile))
        assertFalse("sensor noise must not count as a sign starting", d.hasStartedMotion)
    }
}
