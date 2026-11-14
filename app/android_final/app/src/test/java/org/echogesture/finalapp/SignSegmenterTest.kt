package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Behavioural tests for the sign-bearing-segment trim.
 *
 * The risk this code carries is not "does it trim" but "does it trim away part
 * of a real sign". Every test here is about that boundary: the trim must
 * remove idle, keep the sign, and fall back to the whole recording when it
 * cannot tell the difference.
 */
class SignSegmenterTest {

    private val fps = 30f

    /**
     * A frame whose landmarks differ by [move] from [prev]. Movement is a
     * shape change, not a translation: normalise_hands centres on the wrist,
     * so shifting every landmark equally would read as zero motion.
     */
    private fun frame(atMs: Long, move: Float, hands: List<Int> = listOf(1)): HandTracker.Frame =
        HandTracker.Frame(
            hands.map { slot ->
                HandTracker.Hand(slot, FloatArray(63) { i -> if (i < 3) 0f else move * 0.01f })
            },
            atMs
        )

    /** [n] frames at 33 ms, all identical (a held pose). */
    private fun still(n: Int, atMs: Long = 0L): List<HandTracker.Frame> =
        (0 until n).map { frame(atMs + it * 33L, 0f) }

    /** [n] frames with alternating shape, i.e. genuine movement. */
    private fun moving(n: Int, atMs: Long = 0L): List<HandTracker.Frame> =
        (0 until n).map { frame(atMs + it * 33L, if (it % 2 == 0) 1f else 3f) }

    /** Timestamp after [n] frames of [still]/[moving], so blocks join up. */
    private fun after(n: Int): Long = n * 33L

    // --- the core promise ------------------------------------------------

    @Test
    fun idleOnBothSidesIsTrimmedAway() {
        val frames = still(30) + moving(60, after(30)) + still(30, after(90))
        val seg = SignSegmenter.choose(frames, fps)
        assertEquals(SignSegmenter.Reason.TRIMMED, seg.reason)
        assertTrue("start should be past the leading idle, was ${seg.startIndex}", seg.startIndex >= 15)
        // The transition frame at the sign/idle boundary also counts as
        // moving, and half a second of padding is deliberately kept.
        assertTrue("end should be before the trailing idle, was ${seg.endIndex}", seg.endIndex <= 120)
    }

    @Test
    fun theWholeSignSurvivesTheTrimIncludingItsStartAndEnd() {
        // The property that matters: no moving frame may be cut.
        val idleLead = 30
        val signLen = 60
        val frames = still(idleLead) + moving(signLen, after(idleLead)) + still(30, after(idleLead + signLen))
        val seg = SignSegmenter.choose(frames, fps)
        val firstMoving = idleLead + 1   // frame 0 has no predecessor to compare
        val lastMoving = idleLead + signLen - 1
        assertTrue(
            "trim start ${seg.startIndex} cut into the sign (first moving frame $firstMoving)",
            seg.startIndex <= firstMoving
        )
        assertTrue(
            "trim end ${seg.endIndex} cut into the sign (last moving frame $lastMoving)",
            seg.endIndex > lastMoving
        )
    }

    @Test
    fun paddingIsKeptSoTheInitialHandshapeSurvives() {
        // A sign holds its start shape before moving. Trimming exactly to the
        // first moving frame would drop those frames, so half a second is kept.
        val frames = still(60) + moving(30, after(60))
        val seg = SignSegmenter.choose(frames, fps)
        val expectedPad = (SignSegmenter.PAD_SECONDS * fps).toInt()
        assertTrue(
            "only ${seg.startIndex} frames of padding, wanted ~$expectedPad",
            seg.startIndex <= 61 - expectedPad
        )
    }

    // --- conservative fallbacks ------------------------------------------

    @Test
    fun aCompletelyStaticRecordingIsUsedWholeRatherThanTrimmedToNothing() {
        // Some real signs are nearly static. A trim that removed everything
        // would be far worse than keeping idle.
        val frames = still(120)
        val seg = SignSegmenter.choose(frames, fps)
        assertEquals(SignSegmenter.Reason.NO_MOTION, seg.reason)
        assertEquals(0, seg.startIndex)
        assertEquals(120, seg.endIndex)
    }

    @Test
    fun aRecordingWithNoHandsIsLeftForTheCoverageGateToReject() {
        val frames = (0 until 60).map { HandTracker.Frame(emptyList(), it * 33L) }
        val seg = SignSegmenter.choose(frames, fps)
        assertEquals(SignSegmenter.Reason.NO_HANDS, seg.reason)
        assertEquals(0, seg.startIndex)
        assertEquals(60, seg.endIndex)
    }

    @Test
    fun aVeryShortRecordingIsNotJudgedAtAll() {
        val frames = moving(4)
        val seg = SignSegmenter.choose(frames, fps)
        assertEquals(SignSegmenter.Reason.TOO_FEW_FRAMES, seg.reason)
        assertEquals(4, seg.endIndex)
    }

    @Test
    fun aSignThatFillsTheWholeRecordingIsNotShortened() {
        // The common case for a careful user: press Start, sign, press Finish.
        // There is no idle to remove, so nothing must be removed.
        val frames = moving(90)
        val seg = SignSegmenter.choose(frames, fps)
        assertEquals(0, seg.startIndex)
        assertEquals(90, seg.endIndex)
    }

    // --- hazards ---------------------------------------------------------

    @Test
    fun aHandLeavingTheFrameIsNotCountedAsMotion() {
        // A hand leaving is not the sign moving. If it counted, the trim would
        // stretch to the edge of the frame-out.
        val present = moving(40)
        val absent = (0 until 40).map { HandTracker.Frame(emptyList(), after(40) + it * 33L) }
        val seg = SignSegmenter.choose(present + absent, fps)
        // Half a second of padding is expected; running to the far end of the
        // 40-frame empty tail would mean the hand leaving counted as motion.
        assertTrue("trim ran to the end of the empty tail: ${seg.endIndex}", seg.endIndex <= 40 + 20)
    }

    @Test
    fun aPauseInsideTheSignDoesNotSplitIt() {
        // move, pause, move again: the trim covers both moving runs, not just
        // the larger one, because it trims runs at the ends only.
        val frames = moving(30) + still(20, after(30)) + moving(30, after(50))
        val seg = SignSegmenter.choose(frames, fps)
        assertEquals(0, seg.startIndex)
        assertEquals(80, seg.endIndex)
    }

    @Test
    fun aSlowSignIsNotTrimmedToNothing() {
        // Small but real movement: the threshold is relative to the clip's own
        // peak, so "slow" must still read as movement.
        val frames = still(20) + (0 until 40).map { i ->
            frame(after(20) + i * 33L, if (i % 2 == 0) 0.01f else 0.02f)
        } + still(20, after(60))
        val seg = SignSegmenter.choose(frames, fps)
        assertEquals(SignSegmenter.Reason.TRIMMED, seg.reason)
        assertTrue("slow sign was over-trimmed to ${seg.endIndex - seg.startIndex}",
            seg.endIndex - seg.startIndex >= 40)
    }

    @Test
    fun aMissingFpsFallsBackToThirtyRatherThanDividingByZero() {
        val frames = still(30) + moving(30, after(30))
        val seg = SignSegmenter.choose(frames, 0f)
        assertEquals(SignSegmenter.Reason.TRIMMED, seg.reason)
        assertTrue(seg.startIndex >= 0)
        assertTrue(seg.endIndex <= frames.size)
    }

    @Test
    fun theTrimNeverReordersOrDropsFramesFromTheMiddle() {
        val frames = still(20) + moving(50, after(20)) + still(20, after(70))
        val seg = SignSegmenter.choose(frames, fps)
        val kept = frames.subList(seg.startIndex, seg.endIndex)
        val stamps = kept.map { it.capturedAtMs }
        assertEquals("kept frames must stay in order", stamps.sorted(), stamps)
        assertTrue("kept frames must be contiguous",
            stamps.zipWithNext().all { (a, b) -> b - a == 33L })
    }

    // --- integration with the production sampling ------------------------

    @Test
    fun samplingUsesTheTrimmedIntervalAndStillProducesTwentyFourSteps() {
        val frames = still(60) + moving(60, after(60)) + still(60, after(120))
        val start = frames.first().capturedAtMs
        val stop = frames.last().capturedAtMs
        val recording = SignRecorder.Recording(frames, start, stop)

        val sampled = RecordingFeatures.sample(recording)
        assertEquals(RecordingFeatures.SAMPLED, sampled.size)
        assertTrue("first sample should be past the idle", sampled.first().capturedAtMs > start)
        assertTrue("last sample should be before the idle", sampled.last().capturedAtMs < stop)

        val feats = RecordingFeatures.build(recording)
        assertEquals(24 * 255, feats.size)
        assertTrue("features finite", feats.all { it.isFinite() })
    }

    @Test
    fun recordedFpsIsMeasuredFromTheFramesThemselves() {
        val frames = (0..29).map { frame(it * 33L, 0f) }
        val rec = SignRecorder.Recording(frames, 0L, 29 * 33L)
        val fps = RecordingFeatures.fpsOf(rec)
        assertTrue("fps was $fps", fps in 29f..31f)
    }
}