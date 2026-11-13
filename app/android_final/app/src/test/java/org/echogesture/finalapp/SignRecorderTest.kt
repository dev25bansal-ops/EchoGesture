package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Behavioural tests for the explicit recording mode.
 *
 * Every test drives [SignRecorder] and [RecordingFeatures] — the classes the
 * production button handlers call — with frames stamped at chosen instants.
 * The point is to prove the properties the mode is sold on: nothing from
 * before Start, nothing after Stop, and a real interval resampled to the
 * model's 24 steps.
 */
class SignRecorderTest {

    private fun frame(atMs: Long, hands: List<Int> = listOf(1), seed: Float = 0f): HandTracker.Frame =
        HandTracker.Frame(
            hands.map { slot ->
                // Vary the SHAPE, not a uniform offset: normalise_hands centres
                // each hand on its wrist, so translating every landmark by the
                // same amount produces identical features by construction.
                HandTracker.Hand(
                    slot,
                    FloatArray(63) { i ->
                        val base = (i % 21) * 0.01f
                        if (i < 3) base else base + seed * 0.05f
                    }
                )
            },
            atMs
        )

    /** Frames every [stepMs] from [from] to [to] inclusive. */
    private fun stream(from: Long, to: Long, stepMs: Long = 33L, hands: List<Int> = listOf(1)) =
        generateSequence(from) { it + stepMs }.takeWhile { it <= to }
            .map { frame(it, hands) }
            .toList()

    private fun usable(r: SignRecorder.Result): SignRecorder.Recording {
        assertTrue("expected Usable, was $r", r is SignRecorder.Result.Usable)
        return (r as SignRecorder.Result.Usable).recording
    }

    private fun rejection(r: SignRecorder.Result): SignRecorder.Rejection {
        assertTrue("expected Unusable, was $r", r is SignRecorder.Result.Unusable)
        return (r as SignRecorder.Result.Unusable).reason
    }

    // 1. A duplicated static frame is not a dynamic sequence ---------------

    @Test
    fun oneFrameRepeatedAtTheSameTimestampIsRejectedAsTooShort() {
        // The precise anti-pattern the brief forbids: feeding the last photo
        // repeatedly to fake a video. Ten copies of one instant is zero
        // elapsed time, so it must not pass as a sign.
        val rec = SignRecorder()
        rec.start(1_000)
        repeat(10) { rec.offer(frame(1_000)) }
        assertEquals(SignRecorder.Rejection.TOO_SHORT, rejection(rec.stop(1_000)))
    }

    @Test
    fun oneFrameHeldForARealDurationIsStillRejected() {
        // Even with a genuine 2 s interval, a single analysed frame is not a
        // temporal sequence. The gap check catches it because the only frame
        // sits at the start and nothing follows.
        val rec = SignRecorder()
        rec.start(0)
        rec.offer(frame(0))
        assertEquals(SignRecorder.Rejection.TOO_SHORT, rejection(rec.stop(50)))
    }

    // 2. Sampling across the full interval ---------------------------------

    @Test
    fun samplingSpansTheWholeRecordedIntervalNotItsOpeningFrames() {
        val rec = SignRecorder()
        rec.start(0)
        stream(0, 3_000, 33L).forEach { rec.offer(it) }
        val recording = usable(rec.stop(3_000))

        val sampled = RecordingFeatures.sample(recording)
        assertEquals(RecordingFeatures.SAMPLED, sampled.size)
        assertEquals("first sample is the interval start", 0L, sampled.first().capturedAtMs)
        assertEquals(
            "last sample is the newest real frame, not frame 32",
            2_970L,
            sampled.last().capturedAtMs
        )
        val step = sampled[1].capturedAtMs - sampled[0].capturedAtMs
        assertTrue("sample step was $step ms, expected even spacing", step in 80L..120L)
    }

    @Test
    fun everySampleIsARealFrameThatWasCaptured() {
        val rec = SignRecorder()
        rec.start(0)
        val real = stream(0, 2_500, 33L)
        real.forEach { rec.offer(it) }
        val recording = usable(rec.stop(2_500))

        val stamps = real.map { it.capturedAtMs }.toSet()
        RecordingFeatures.sample(recording).forEach {
            assertTrue("invented frame at ${it.capturedAtMs}", it.capturedAtMs in stamps)
        }
    }

    // 3. Start/Stop select exactly the expected interval -------------------

    @Test
    fun startAndStopSelectExactlyTheirInterval() {
        val rec = SignRecorder()
        rec.start(1_000)
        stream(1_000, 2_500, 33L).forEach { rec.offer(it) }
        val recording = usable(rec.stop(2_500))
        assertEquals(1_000L, recording.startedAtMs)
        assertEquals(2_500L, recording.stoppedAtMs)
        assertEquals(1_500L, recording.durationMs)
        assertTrue(recording.frames.first().capturedAtMs >= 1_000)
        assertTrue(recording.frames.last().capturedAtMs <= 2_500)
    }

    // 4. No pre-start contamination ----------------------------------------

    @Test
    fun framesOfferedBeforeStartAreDropped() {
        val rec = SignRecorder()
        // The analyser is running before the press, exactly as in production.
        stream(0, 900, 33L).forEach { assertFalse(rec.offer(it)) }
        rec.start(1_000)
        stream(1_000, 2_000, 33L).forEach { rec.offer(it) }
        val recording = usable(rec.stop(2_000))
        assertTrue(
            "a pre-start frame entered the recording",
            recording.frames.all { it.capturedAtMs >= 1_000 }
        )
    }

    @Test
    fun aPreviousSignCannotLeakIntoTheNextRecording() {
        val rec = SignRecorder()
        rec.start(0)
        stream(0, 2_000, 33L).forEach { rec.offer(it) }
        usable(rec.stop(2_000))
        // Second attempt: the buffer must start empty, not append.
        rec.start(5_000)
        assertEquals(0, rec.frameCount())
        stream(5_000, 6_000, 33L).forEach { rec.offer(it) }
        val second = usable(rec.stop(6_000))
        assertTrue("old frames leaked", second.frames.all { it.capturedAtMs >= 5_000 })
    }

    // 5. No post-stop contamination ----------------------------------------

    @Test
    fun framesArrivingAfterStopAreDropped() {
        val rec = SignRecorder()
        rec.start(1_000)
        stream(1_000, 2_000, 33L).forEach { rec.offer(it) }
        usable(rec.stop(2_000))
        // The analyser keeps delivering; nothing more may be stored.
        assertFalse(rec.offer(frame(2_100)))
        assertEquals(0, rec.frameCount())
    }

    // 6. No stale buffer reuse ---------------------------------------------

    @Test
    fun startingAgainNeverReusesThePreviousFrames() {
        val rec = SignRecorder()
        rec.start(0)
        stream(0, 1_500, 33L).forEach { rec.offer(it) }
        rec.stop(1_500)
        rec.start(10_000)
        assertEquals(0, rec.frameCount())
        assertTrue(rec.isRecording)
    }

    @Test
    fun cancellingDiscardsEverythingAndEndsTheRecording() {
        val rec = SignRecorder()
        rec.start(0)
        stream(0, 1_000, 33L).forEach { rec.offer(it) }
        rec.cancel()
        assertFalse(rec.isRecording)
        assertNull(rec.startedAt())
        assertEquals(SignRecorder.Rejection.NOT_STARTED, rejection(rec.stop(1_000)))
    }

    // 7. Monotonic timestamps ----------------------------------------------

    @Test
    fun theRecordingPreservesArrivalOrderAndMonotonicTimestamps() {
        val rec = SignRecorder()
        rec.start(0)
        stream(0, 2_000, 33L).forEach { rec.offer(it) }
        val recording = usable(rec.stop(2_000))
        val stamps = recording.frames.map { it.capturedAtMs }
        assertEquals("order must be preserved", stamps.sorted(), stamps)
        stamps.zipWithNext().forEach { (a, b) -> assertTrue("non-monotonic $a -> $b", b >= a) }
    }

    @Test
    fun aFrameStampedOutsideTheIntervalIsExcludedAtStop() {
        // Belt-and-braces: offer() already refuses, but stop() re-checks, so a
        // caller that bypassed offer() still cannot smuggle a frame in.
        val rec = SignRecorder()
        rec.start(1_000)
        stream(1_000, 2_000, 33L).forEach { rec.offer(it) }
        val recording = usable(rec.stop(2_000))
        assertTrue(recording.frames.all { it.capturedAtMs in 1_000..2_000 })
    }

    // 8. Low hand coverage -------------------------------------------------

    @Test
    fun aRecordingWithAlmostNoHandsIsRejectedAsLowCoverage() {
        val rec = SignRecorder()
        rec.start(0)
        // 3.7 s of frames, only ~3% with a hand.
        stream(0, 3_700, 33L, hands = emptyList()).forEach { rec.offer(it) }
        rec.offer(frame(3_700, listOf(1)))
        val r = rec.stop(3_700)
        assertEquals(SignRecorder.Rejection.LOW_COVERAGE, rejection(r))
    }

    // 9. No hands ----------------------------------------------------------

    @Test
    fun aRecordingWithNoHandsAtAllIsRejected() {
        val rec = SignRecorder()
        rec.start(0)
        stream(0, 2_000, 33L, hands = emptyList()).forEach { rec.offer(it) }
        assertEquals(SignRecorder.Rejection.LOW_COVERAGE, rejection(rec.stop(2_000)))
    }

    // 10. A brief sign -----------------------------------------------------

    @Test
    fun aBriefSignIsAcceptedAndResampledToTwentyFourSteps() {
        val rec = SignRecorder()
        rec.start(0)
        stream(0, 600, 33L).forEach { rec.offer(it) }
        val recording = usable(rec.stop(600))
        val features = RecordingFeatures.build(recording)
        assertEquals(
            ModelContract.EXPECTED_FRAMES * ModelContract.EXPECTED_FEATURE_DIM,
            features.size
        )
    }

    // 11. A long sign ------------------------------------------------------

    @Test
    fun aLongSignIsAcceptedAndStillProducesTwentyFourSteps() {
        val rec = SignRecorder()
        rec.start(0)
        stream(0, 9_000, 33L).forEach { rec.offer(it) }
        val recording = usable(rec.stop(9_000))
        assertEquals(
            ModelContract.EXPECTED_FRAMES * ModelContract.EXPECTED_FEATURE_DIM,
            RecordingFeatures.build(recording).size
        )
    }

    @Test
    fun aSignPastTheCeilingAutoStopsRatherThanGrowingUnbounded() {
        val rec = SignRecorder()
        rec.start(0)
        assertFalse(rec.shouldAutoStop(SignRecorder.MAX_DURATION_MS - 1))
        assertTrue(rec.shouldAutoStop(SignRecorder.MAX_DURATION_MS + 1))
    }

    @Test
    fun hittingTheFrameCapTruncatesTheEndNotTheBeginning() {
        val rec = SignRecorder(maxFrames = 10)
        rec.start(0)
        stream(0, 2_000, 33L).forEach { rec.offer(it) }
        assertTrue(rec.wasTruncated())
        assertEquals(10, rec.frameCount())
        // The frames kept are the first ten: the sign's onset survives.
        val kept = rec.stop(2_000)
        assertTrue(kept is SignRecorder.Result.Unusable)
        val frames = (kept as SignRecorder.Result.Unusable).recording!!.frames
        assertEquals(0L, frames.first().capturedAtMs)
        assertEquals(9 * 33L, frames.last().capturedAtMs)
    }

    // 12. Pauses during a sign ---------------------------------------------

    @Test
    fun aBriefPauseInsideASignDoesNotInvalidateIt() {
        // A held hand still produces frames; only the features go flat. The
        // recorder must not treat "no movement" as "no capture".
        val rec = SignRecorder()
        rec.start(0)
        stream(0, 1_000, 33L).forEach { rec.offer(it) }
        // 300 ms of genuinely identical hand poses.
        (0..9).forEach { rec.offer(frame(1_033 + it * 33L, listOf(1), seed = 0f)) }
        stream(1_400, 2_500, 33L).forEach { rec.offer(it) }
        val recording = usable(rec.stop(2_500))
        assertTrue(recording.frameCount > 40)
    }

    @Test
    fun aLongCameraStallMidRecordingIsRejected() {
        val rec = SignRecorder()
        rec.start(0)
        stream(0, 800, 33L).forEach { rec.offer(it) }
        // 1.5 s hole, then frames resume: the sequence has a break in it.
        stream(2_300, 3_000, 33L).forEach { rec.offer(it) }
        assertEquals(SignRecorder.Rejection.CAMERA_STALLED, rejection(rec.stop(3_000)))
    }

    // 13. Camera interruption ----------------------------------------------

    @Test
    fun aRecordingCutOffByACameraSwitchYieldsNoFramesRatherThanMixedOnes() {
        val rec = SignRecorder()
        rec.start(0)
        stream(0, 1_000, 33L).forEach { rec.offer(it) }
        // clear() on camera switch cancels the recording.
        rec.cancel()
        assertEquals(SignRecorder.Rejection.NOT_STARTED, rejection(rec.stop(1_500)))
    }

    // 15. Repeated capture attempts ----------------------------------------

    @Test
    fun threeConsecutiveRecordingsStayIndependent() {
        val rec = SignRecorder()
        val results = (0 until 3).map { i ->
            val base = i * 10_000L
            rec.start(base)
            stream(base, base + 1_200, 33L).forEach { rec.offer(it) }
            usable(rec.stop(base + 1_200))
        }
        assertEquals(3, results.size)
        results.forEachIndexed { i, r ->
            assertTrue(
                "recording $i leaked frames from another attempt",
                r.frames.all { it.capturedAtMs in (i * 10_000L)..(i * 10_000L + 1_200) }
            )
        }
    }

    // 16. Correct 24x255 shape and motion integrity ------------------------

    @Test
    fun theTensorIsTwentyFourByTwoFiftyFive() {
        val rec = SignRecorder()
        rec.start(0)
        stream(0, 3_000, 33L).forEach { rec.offer(it) }
        val features = RecordingFeatures.build(usable(rec.stop(3_000)))
        assertEquals(24 * 255, features.size)
        assertEquals(255, ModelContract.EXPECTED_FEATURE_DIM)
        assertEquals(24, ModelContract.EXPECTED_FRAMES)
    }

    @Test
    fun realMovementProducesNonzeroVelocityAcrossTheSteps() {
        // Distinct hand poses per frame: the model must be told the hands
        // moved, not merely that frames arrived.
        val rec = SignRecorder()
        rec.start(0)
        (0..90).forEach { i ->
            rec.offer(frame(i * 33L, listOf(1), seed = i * 3f))
        }
        val features = RecordingFeatures.build(usable(rec.stop(91 * 33L)))
        assertTrue("velocity features were all zero", RecordingFeatures.meanMotion(features) > 0f)
        assertTrue(
            "no step differed from its neighbour",
            RecordingFeatures.movingSteps(features) > 0
        )
    }

    @Test
    fun aHeldStaticPoseIsNotRejectedForLowMotion() {
        // Some real signs are nearly static. Capture integrity is about
        // timestamps and coverage, never about how much the hand moved.
        val rec = SignRecorder()
        rec.start(0)
        (0..90).forEach { i -> rec.offer(frame(i * 33L, listOf(1), seed = 0f)) }
        val recording = usable(rec.stop(91 * 33L))
        val features = RecordingFeatures.build(recording)
        assertEquals(0f, RecordingFeatures.meanMotion(features), 1e-6f)
        assertEquals(0, RecordingFeatures.movingSteps(features))
    }

    @Test
    fun noHandIntervalsArePreservedRatherThanFilled() {
        val rec = SignRecorder()
        rec.start(0)
        (0..90).forEach { i ->
            val hands = if (i in 20..40) emptyList() else listOf(1)
            rec.offer(frame(i * 33L, hands))
        }
        val recording = usable(rec.stop(91 * 33L))
        val sampled = RecordingFeatures.sample(recording)
        assertTrue(
            "empty frames were filled in",
            sampled.any { it.hands.isEmpty() } && sampled.any { it.hands.isNotEmpty() }
        )
    }

    @Test
    fun startReturnsTheInstantTheRecordingBegan() {
        val rec = SignRecorder()
        rec.start(4_242)
        assertNotNull(rec.startedAt())
        assertEquals(4_242L, rec.startedAt())
    }
}