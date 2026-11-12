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
        // The capture must be long enough to hold a dynamic sign yet bounded so
        // a stalled analyser cannot keep it open forever.
        assertTrue(HandTracker.RING_CAPACITY >= 24)
        assertTrue(HandTracker.MIN_CAPTURE_MS <= HandTracker.MAX_CAPTURE_MS)
        assertTrue(HandTracker.MAX_CAPTURE_FRAMES >= HandTracker.RING_CAPACITY)
    }
}