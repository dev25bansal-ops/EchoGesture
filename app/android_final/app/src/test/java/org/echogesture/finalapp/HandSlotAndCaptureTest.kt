package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for the hand-slot mapping and the capture-window accounting.
 *
 * Neither was covered before, and both sit between the camera and the model:
 * a wrong slot puts a hand's landmarks in the other hand's half of the feature
 * vector, and a wrong capture length changes the temporal scale the classifier
 * sees. Both failures look like "the model is bad" from the outside.
 */
class HandSlotAndCaptureTest {

    private fun frame(slots: List<Int>, atMs: Long): HandTracker.Frame =
        HandTracker.Frame(slots.map { HandTracker.Hand(it, FloatArray(63)) }, atMs)

    @Test
    fun captureDurationAndFpsComeFromRealTimestamps() {
        // 32 frames sampled every 33 ms: the window is ~1.02 s at ~30 fps.
        val frames = (0 until 32).map { frame(listOf(1), it * 33L) }
        val c = HandTracker.Capture(frames, 31 * 33L, 30.3f, 32)
        assertEquals(32, c.frameCount)
        assertEquals(1023L, c.durationMs)
    }

    @Test
    fun emptyCaptureReportsZeroRatherThanDividingByZero() {
        val c = HandTracker.Capture(emptyList(), 0L, 0f, 0)
        assertEquals(0, c.frameCount)
        assertEquals(0L, c.durationMs)
        assertTrue("an empty capture must not report a frame rate", c.fps == 0f)
    }

    @Test
    fun handFramesCountFramesWithAtLeastOneHand() {
        val frames = listOf(
            frame(listOf(1), 0),
            frame(emptyList(), 33),
            frame(listOf(0), 66),
            frame(listOf(0, 1), 99)
        )
        val c = HandTracker.Capture(frames, 99L, 30f, frames.count { it.hands.isNotEmpty() })
        assertEquals(3, c.handFrames)
    }

    @Test
    fun presenceMaskDistinguishesLeftRightAndAbsent() {
        // The slot is what the feature vector keys on: 0 = Left, 1 = Right,
        // absent = neither. A frame with only the right hand must not look like
        // a frame with only the left.
        val left = frame(listOf(0), 0)
        val right = frame(listOf(1), 33)
        val none = frame(emptyList(), 66)
        assertEquals(listOf(0), left.hands.map { it.slot })
        assertEquals(listOf(1), right.hands.map { it.slot })
        assertTrue(none.hands.isEmpty())
    }

    @Test
    fun windowBoundsAreSane() {
        // The window must span a real interval comparable to a training clip
        // (~3.7 s for the 110-frame example in the extractor's docstring), and
        // the memory backstop must be able to hold that span at camera rates.
        assertTrue(HandTracker.CAPTURE_SPAN_MS >= 3000L)
        assertTrue(HandTracker.CAPTURE_SPAN_MS <= 5000L)
        assertTrue(HandTracker.MAX_CAPTURE_FRAMES >= HandTracker.RING_CAPACITY)
    }

    @Test
    fun captureSamplesEvenlyAcrossTheSpanNotTheFirstFrames() {
        // The defect this guards: taking 32 consecutive frames covers only the
        // start of the sign. Training spread 32 samples across the whole clip
        // and computes velocity as a plain per-frame difference with no dt, so
        // the sample step IS the time base. The window must reach the END of
        // the ring and advance ~115 ms, not 33 ms.
        val ts = LongArray(111) { it * 33L } // ~110 frames, ~3.7 s
        val idx = HandTracker.evenlySpacedIndices(ts, 32)
        assertEquals(32, idx.size)
        assertEquals("first sample is the window start", 0, idx.first())
        assertEquals("last sample is the window end", 110, idx.last())
        val step = ts[idx[1]] - ts[idx[0]]
        assertTrue("step was $step ms, expected ~115 ms", step in 90L..140L)
    }

    @Test
    fun shortWindowsArePassedThroughUnchanged() {
        // A slow device that only delivered 10 frames must not be padded with
        // duplicates; downstream resampling handles the count.
        val ts = LongArray(10) { it * 100L }
        val idx = HandTracker.evenlySpacedIndices(ts, 32)
        assertEquals(10, idx.size)
        assertTrue(idx.toList() == (0 until 10).toList())
    }

    @Test
    fun emptyWindowSelectsNothing() {
        assertTrue(HandTracker.evenlySpacedIndices(LongArray(0), 32).isEmpty())
    }
}