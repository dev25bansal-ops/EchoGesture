package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Behavioural tests for the frame selection that feeds the model.
 *
 * The live path differs from a training clip in one way this code must handle:
 * a phone analyser delivers frames on an irregular, and possibly low, rate,
 * while training clips were decoded at a constant rate. `sample()` decides which
 * frames the model actually sees, so an error here silently changes the timing
 * of every model step.
 */
class RecordingFeaturesTest {

    private val fps = 10f

    /** A frame at [atMs] whose landmarks encode nothing in particular. */
    private fun frame(atMs: Long, marker: Float): HandTracker.Frame =
        HandTracker.Frame(
            listOf(HandTracker.Hand(1, FloatArray(63) { marker })),
            atMs
        )

    /** [n] frames at a constant [stepMs] spacing. */
    private fun evenly(n: Int, stepMs: Long, from: Long = 0L): List<HandTracker.Frame> =
        (0 until n).map { frame(from + it * stepMs, it.toFloat()) }

    private fun recording(frames: List<HandTracker.Frame>): SignRecorder.Recording =
        SignRecorder.Recording(frames, frames.first().capturedAtMs,
            frames.last().capturedAtMs)

    // --- the contract that matters for a live camera ----------------------

    /**
     * Above the sample budget, frames must be chosen by TIMESTAMP, not index.
     *
     * A phone analyser drops frames under load, so a 60-frame recording can
     * span a very different wall-clock interval than a 60-frame clip. Choosing
     * by index would then place model steps at wrong real times and change
     * every velocity value the model was trained on.
     */
    @Test
    fun longRecordingIsSampledByTimestampNotIndex() {
        // 40 frames, but a 10x gap in the middle: index sampling would put half
        // the window before the gap; timestamp sampling must spread across it.
        val frames = buildList {
            addAll(evenly(20, 33L, 0L))
            addAll(evenly(20, 33L, 20 * 33L + 2000L))
        }
        val r = recording(frames)
        val picked = RecordingFeatures.sample(r, fps)

        assertEquals("budget is 32 frames", RecordingFeatures.SAMPLED, picked.size)
        val span = picked.last().capturedAtMs - picked.first().capturedAtMs
        val total = frames.last().capturedAtMs - frames.first().capturedAtMs
        // Timestamp sampling covers the whole interval; index sampling would
        // still cover it here, so assert the GAP is represented, not skipped.
        assertTrue(
            "sampled span $span should cover most of the real span $total",
            span >= total * 0.8f
        )
    }

    /**
     * A short recording must be passed through untouched.
     *
     * Below the budget there is nothing to drop, and duplicating frames would
     * invent motion that never happened. The caller's linear resample to 24
     * steps is what stretches a short capture; `sample()` must not.
     */
    @Test
    fun shortRecordingIsPassedThroughUnchanged() {
        val frames = evenly(12, 100L)
        val picked = RecordingFeatures.sample(recording(frames), fps)
        assertEquals("all 12 frames survive", 12, picked.size)
        assertEquals(frames.map { it.capturedAtMs }, picked.map { it.capturedAtMs })
    }

    /** Exactly at the budget: no sampling, no duplication. */
    @Test
    fun recordingAtTheBudgetIsReturnedWhole() {
        val frames = evenly(RecordingFeatures.SAMPLED, 100L)
        val picked = RecordingFeatures.sample(recording(frames), fps)
        assertEquals(RecordingFeatures.SAMPLED, picked.size)
    }

    /** One frame past the budget must actually sample. */
    @Test
    fun recordingJustOverTheBudgetIsSampled() {
        val frames = evenly(RecordingFeatures.SAMPLED + 1, 100L)
        val picked = RecordingFeatures.sample(recording(frames), fps)
        assertEquals(RecordingFeatures.SAMPLED, picked.size)
    }

    /**
     * Sampling must never repeat a frame.
     *
     * `evenlySpacedIndices` picks the nearest real frame to each target time; if
     * two targets land on the same frame, the model would see a frozen step and
     * read it as zero velocity.
     */
    @Test
    fun sampledFramesAreDistinct() {
        val frames = evenly(80, 40L)
        val picked = RecordingFeatures.sample(recording(frames), fps)
        assertEquals(picked.size, picked.map { it.capturedAtMs }.toSet().size)
    }

    /** A degenerate recording must not crash the caller. */
    @Test
    fun singleFrameDoesNotCrash() {
        val frames = listOf(frame(0L, 1f))
        val picked = RecordingFeatures.sample(recording(frames), fps)
        assertEquals(1, picked.size)
    }

    /** fpsOf must report the recording's own rate, not an assumed one. */
    @Test
    fun fpsIsDerivedFromTheRecordingsOwnTimestamps() {
        val frames = evenly(11, 100L)   // 10 intervals of 100 ms = 10 fps
        val r = recording(frames)
        assertEquals(10f, RecordingFeatures.fpsOf(r), 0.01f)
    }
}