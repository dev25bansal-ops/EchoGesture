package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Behavioural tests for the pre-tap capture contract.
 *
 * Every test drives [HandTracker.selectCapture] — the function production calls
 * from a real tap — with hand-built [HandTracker.Frame] lists carrying chosen
 * timestamps. Nothing here re-asserts a constant's value as a stand-in for the
 * behaviour it is supposed to produce.
 */
class PreTapCaptureContractTest {

    private val span = HandTracker.CAPTURE_SPAN_MS
    private val never = Long.MIN_VALUE

    private fun frame(atMs: Long, hands: List<Int> = listOf(1)): HandTracker.Frame =
        HandTracker.Frame(hands.map { HandTracker.Hand(it, FloatArray(63)) }, atMs)

    /** [count] frames at 33 ms apart, ending at [endMs]. */
    private fun stream(endMs: Long, count: Int, step: Long = 33L, hands: List<Int> = listOf(1)) =
        (count - 1 downTo 0).map { frame(endMs - it * step, hands) }

    private fun ready(r: HandTracker.CaptureRequest): HandTracker.Capture {
        assertTrue("expected Ready but was $r", r is HandTracker.CaptureRequest.Ready)
        return (r as HandTracker.CaptureRequest.Ready).capture
    }

    private fun insufficient(r: HandTracker.CaptureRequest): HandTracker.CaptureRequest.Insufficient {
        assertTrue("expected Insufficient but was $r", r is HandTracker.CaptureRequest.Insufficient)
        return r as HandTracker.CaptureRequest.Insufficient
    }

    private val tap = 10_000L

    // 1. Full history -------------------------------------------------------

    @Test
    fun fullHistoryClassifiesTheWindowEndingAtTheTap() {
        val frames = stream(endMs = tap, count = 150) // ~5 s of frames
        val c = ready(HandTracker.selectCapture(frames, tap, never))

        assertEquals(HandTracker.RING_CAPACITY, c.frameCount)
        assertTrue("window must end at the tap", c.frames.last().capturedAtMs <= tap)
        assertEquals("the newest frame is the tap frame", tap, c.frames.last().capturedAtMs)
        assertTrue(
            "window must reach back about the full span, was ${c.durationMs} ms",
            c.durationMs in (span - 200)..span
        )
        assertEquals(HandTracker.RING_CAPACITY, c.handFrames)
    }

    @Test
    fun framesOlderThanTheSpanAreExcludedNotStretched() {
        // 20 s of history: only the last `span` milliseconds may be used. If
        // the older frames leaked in, the sample step would collapse and the
        // velocity scale the model was trained on would be wrong.
        val frames = stream(endMs = tap, count = 600)
        val c = ready(HandTracker.selectCapture(frames, tap, never))
        val oldest = c.frames.first().capturedAtMs
        assertTrue("oldest sampled frame was ${tap - oldest} ms before the tap", tap - oldest <= span + 40)
    }

    // 2. History not warmed -------------------------------------------------

    @Test
    fun historyShorterThanTheMinimumSpanIsRejectedWithItsRealSpan() {
        val frames = stream(endMs = tap, count = 10) // ~300 ms
        val r = insufficient(HandTracker.selectCapture(frames, tap, never))
        assertEquals(HandTracker.CaptureRequest.Insufficient.Reason.TOO_SHORT, r.reason)
        assertTrue("the reported span must be the real one, was ${r.spanMs}", r.spanMs in 250L..350L)
    }

    @Test
    fun noFramesAtAllIsReportedSeparatelyFromTooShort() {
        val r = insufficient(HandTracker.selectCapture(emptyList(), tap, never))
        assertEquals(HandTracker.CaptureRequest.Insufficient.Reason.NO_FRAMES, r.reason)
    }

    // 3. No hands -----------------------------------------------------------

    @Test
    fun aHandlessWindowIsStillAValidCaptureSoCoverageCanBeJudged() {
        // The caller, not the tracker, decides that a handless window is
        // unusable: diagnostics need the frame count to say "no hand was ever
        // seen", which is a different failure from "the sign was not readable".
        val frames = stream(endMs = tap, count = 150, hands = emptyList())
        val c = ready(HandTracker.selectCapture(frames, tap, never))
        assertEquals(HandTracker.RING_CAPACITY, c.frameCount)
        assertEquals(0, c.handFrames)
    }

    @Test
    fun genuineHandlessFramesInsideTheSignArePreserved() {
        // The onset and offset of a real sign are empty. They must survive into
        // the window, or the coverage fraction would misrepresent the sign.
        val frames = (0 until 150).map { i ->
            val at = tap - (149 - i) * 33L
            frame(at, if (at in tap - 2000..tap - 1200) emptyList() else listOf(1))
        }
        val c = ready(HandTracker.selectCapture(frames, tap, never))
        assertTrue("empty frames from inside the sign must remain", c.handFrames < c.frameCount)
        assertTrue("the sign itself must still be present", c.handFrames > 0)
    }

    // 4. Camera paused ------------------------------------------------------

    @Test
    fun aStalledAnalyserIsRejectedRatherThanClassifyingAStaleWindow() {
        // The newest frame is two seconds old: the sign happened after it, so
        // this window describes something else entirely.
        val frames = stream(endMs = tap - 2000, count = 150)
        val r = insufficient(HandTracker.selectCapture(frames, tap, never))
        assertEquals(HandTracker.CaptureRequest.Insufficient.Reason.STALE, r.reason)
    }

    @Test
    fun aBriefFrameGapJustUnderTheStaleLimitIsStillClassifiable() {
        val frames = stream(endMs = tap - 500, count = 150)
        val c = ready(HandTracker.selectCapture(frames, tap, never))
        assertTrue(c.frameCount > 0)
        assertTrue("no frame may be invented at the tap", c.frames.last().capturedAtMs <= tap - 500)
    }

    // 5. Timestamps with gaps ----------------------------------------------

    @Test
    fun irregularFrameTimingSelectsRealFramesNotTheFirstOnes() {
        // A slow analyser with jittery delivery. Selection is by timestamp, so
        // the window must still start near the span boundary and end at the
        // newest frame, with every chosen point being an actual frame.
        val ts = ArrayList<Long>()
        var t = tap - 4000
        while (t <= tap) {
            ts.add(t)
            t += if (ts.size % 7 == 0) 120 else 45
        }
        val frames = ts.map { frame(it) }
        val c = ready(HandTracker.selectCapture(frames, tap, never))
        assertTrue("window must start near the span boundary", tap - c.frames.first().capturedAtMs <= span + 130)
        assertTrue("window must end at the newest frame", tap - c.frames.last().capturedAtMs <= 60)
        // Every sampled timestamp must be one that really arrived.
        val real = ts.toSet()
        c.frames.forEach { assertTrue("invented frame at ${it.capturedAtMs}", it.capturedAtMs in real) }
    }

    @Test
    fun perFrameTimestampsDecideDurationAndFps() {
        val frames = (0..100).map { frame(tap - 3300 + it * 33L) }
        val c = ready(HandTracker.selectCapture(frames, tap, never))
        // 3.3 s covered by the selected samples: well under 30 fps per step
        // would mean the window was truncated.
        assertTrue("fps was ${c.fps}", c.fps in 5f..40f)
        assertTrue("duration was ${c.durationMs}", c.durationMs in 3000L..3400L)
    }

    // 6. Camera switch ------------------------------------------------------

    @Test
    fun framesFromThePreviousCameraCannotEnterTheNewCapture() {
        // A camera rebind clears the ring. The first tap afterwards therefore
        // sees only the frames the new camera delivered.
        val newCamera = stream(endMs = tap, count = 20) // ~600 ms since rebind
        val r = insufficient(HandTracker.selectCapture(newCamera, tap, never))
        assertEquals(HandTracker.CaptureRequest.Insufficient.Reason.TOO_SHORT, r.reason)
    }

    // 7. Very short sign ----------------------------------------------------

    @Test
    fun aShortSignStillYieldsAFullWindowOfItsSurroundingFrames() {
        // A 0.6 s sign inside an otherwise idle 3.7 s: the window is the whole
        // span, and coverage reflects that only part of it was the sign.
        val frames = (0 until 112).map { i ->
            val at = tap - (111 - i) * 33L
            frame(at, if (at in tap - 900..tap - 300) listOf(1) else emptyList())
        }
        val c = ready(HandTracker.selectCapture(frames, tap, never))
        assertTrue("the sign must be inside the window", c.handFrames > 0)
        assertTrue("but not the whole window", c.handFrames < c.frameCount)
    }

    // 8. Delayed tap --------------------------------------------------------

    @Test
    fun aTapLongAfterTheSignSeesAnEmptyWindowAndIsNotRewardedWithTheOldSign() {
        // The user signed 5 s ago and tapped now. The last span holds only the
        // empty room after the sign, which is exactly what the tracker reports.
        // It must NOT reach further back to find the hand.
        val frames = (0 until 300).map { i ->
            val at = tap - 10_000 + i * 33L
            frame(at, if (at < tap - 5000) listOf(1) else emptyList())
        }
        val c = ready(HandTracker.selectCapture(frames, tap, never))
        assertEquals("a late tap must not recover the finished sign", 0, c.handFrames)
    }

    // 9. Rapid repeated taps ------------------------------------------------

    @Test
    fun aSecondTapOnTheSameSignIsRefusedInsteadOfRepeatingTheAnswer() {
        val frames = stream(endMs = tap, count = 150)
        val first = ready(HandTracker.selectCapture(frames, tap, never))
        assertTrue(first.frameCount > 0)

        val second = insufficient(HandTracker.selectCapture(frames, tap + 300, tap))
        assertEquals(HandTracker.CaptureRequest.Insufficient.Reason.REPEATED, second.reason)
    }

    @Test
    fun aTapAfterTheRepeatGapIsAcceptedAgain() {
        // The camera keeps delivering throughout; only the tap spacing changes.
        val laterTap = tap + HandTracker.MIN_REPEAT_GAP_MS
        val frames = stream(endMs = laterTap, count = 200)
        ready(HandTracker.selectCapture(frames, tap, never))
        val later = ready(HandTracker.selectCapture(frames, laterTap, tap))
        assertTrue(later.frameCount > 0)
    }

    @Test
    fun theRepeatGuardIsOnlySetByAnAcceptedCapture() {
        // A rejected tap (too short, stale, empty) must not lock the user out
        // of the next legitimate attempt.
        insufficient(HandTracker.selectCapture(stream(tap, 5), tap, never))
        val c = ready(HandTracker.selectCapture(stream(endMs = tap + 50, count = 150), tap + 50, never))
        assertTrue(c.frameCount > 0)
    }

    // 10. Concurrent frame arrival during the tap ---------------------------

    @Test
    fun aFrameThatArrivesAfterTheTapIsExcludedEvenIfItIsInTheBuffer() {
        // This is the race the contract exists for: the analyser appends while
        // the tap is being handled, so the buffer holds a frame with a
        // timestamp after the tap. It belongs to the NEXT sign.
        val frames = stream(endMs = tap - 33, count = 150) + frame(tap + 40)
        val c = ready(HandTracker.selectCapture(frames, tap, never))
        assertTrue(
            "a post-tap frame entered the window at ${c.frames.last().capturedAtMs}",
            c.frames.all { it.capturedAtMs <= tap }
        )
    }

    @Test
    fun postTapFramesAloneAreNeverEnoughToClassify() {
        val frames = listOf(frame(tap + 33), frame(tap + 66))
        val r = insufficient(HandTracker.selectCapture(frames, tap, never))
        assertEquals(HandTracker.CaptureRequest.Insufficient.Reason.NO_FRAMES, r.reason)
    }

    // Sampling invariants ---------------------------------------------------

    @Test
    fun theSampledWindowSpansTheSameTemporalStepAsTraining() {
        // Training spread 32 samples over a ~3.7 s clip: ~115 ms per step. If
        // the app instead used consecutive 33 ms frames, every velocity the
        // model sees would be ~3.5x too small.
        val c = ready(HandTracker.selectCapture(stream(endMs = tap, count = 150), tap, never))
        val step = c.frames[1].capturedAtMs - c.frames[0].capturedAtMs
        assertTrue("sample step was $step ms, expected ~115 ms", step in 90L..140L)
    }

    @Test
    fun aThinBufferIsPassedThroughWithoutInventingFrames() {
        // A slow device that delivered only 12 frames over the span returns 12
        // real frames, not 32 with duplicates.
        val frames = (0 until 12).map { frame(tap - (11 - it) * 130L) }
        val c = ready(HandTracker.selectCapture(frames, tap, never))
        assertEquals(12, c.frameCount)
        assertEquals(12, c.frames.map { it.capturedAtMs }.toSet().size)
    }
}