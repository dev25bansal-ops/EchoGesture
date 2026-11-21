package org.echogesture.finalapp

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The one-tap contract, exercised through the real [HandTracker].
 *
 * `AutoSegmenterTest` proves the state machine in isolation. This proves the
 * tracker actually drives it: that arming is reachable, that a completed sign
 * becomes an outcome exactly once, and that leaving the tab or starting a new
 * attempt cannot leak frames from one sign into the next.
 *
 * No camera is involved. These feed synthetic [HandTracker.Frame]s straight into
 * the tracker's segmenter, which is what the analyser does with real ones.
 */
@RunWith(AndroidJUnit4::class)
class OneTapCaptureTest {

    private fun frame(atMs: Long, move: Float, hands: Int = 1): HandTracker.Frame =
        HandTracker.Frame(
            (0 until hands).map { slot ->
                HandTracker.Hand(slot, FloatArray(63) { i -> if (i < 3) 0f else move * 0.01f })
            },
            atMs
        )

    /** A tracker with no landmarker: enough for the segmenter paths under test. */
    private fun tracker(): HandTracker {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        return HandTracker.createForTest(ctx)
    }

    /**
     * Tap, sign, stop — with no second tap.
     *
     * This is the behaviour the brief requires: one press, an automatic end, and
     * an outcome that can be classified.
     */
    @Test
    fun oneTapProducesAnOutcomeWithNoSecondTap() {
        val t = tracker()
        try {
            t.armAuto(0L)
            assertEquals(AutoSegmenter.State.ARMED, t.autoState())

            // Settle, then sign with clear motion, then hold still.
            var outcome: AutoSegmenter.Outcome? = null
            var now = 0L
            for (i in 0 until 12) { now += 40; t.offerForTest(frame(now, 0f)) }
            for (i in 0 until 40) {
                now += 40
                t.offerForTest(frame(now, if (i % 2 == 0) 2f else 6f))
                if (outcome == null) outcome = t.takeAutoOutcome()
            }
            for (i in 0 until 60) {
                now += 40
                t.offerForTest(frame(now, 0f))
                if (outcome == null) outcome = t.takeAutoOutcome()
            }

            assertNotNull("the sign must complete without a second tap", outcome)
            assertTrue("segment must be usable", outcome!!.isUsable)
            assertTrue("segment must contain frames", outcome!!.frameCount > 0)
        } finally {
            t.close()
        }
    }

    /**
     * The outcome is handed over exactly once.
     *
     * A second read must return null; otherwise the UI thread and a late
     * analyser callback could both classify the same sign.
     */
    @Test
    fun completedOutcomeIsConsumedOnce() {
        val t = tracker()
        try {
            t.armAuto(0L)
            var now = 0L
            for (i in 0 until 12) { now += 40; t.offerForTest(frame(now, 0f)) }
            for (i in 0 until 40) {
                now += 40; t.offerForTest(frame(now, if (i % 2 == 0) 2f else 6f))
            }
            for (i in 0 until 60) { now += 40; t.offerForTest(frame(now, 0f)) }

            val first = t.takeAutoOutcome()
            assertNotNull("first read must return the outcome", first)
            assertNull("second read must be empty", t.takeAutoOutcome())
        } finally {
            t.close()
        }
    }

    /** A new attempt must not inherit frames from the previous sign. */
    @Test
    fun reArmingDiscardsThePreviousSignsFrames() {
        val t = tracker()
        try {
            t.armAuto(0L)
            var now = 0L
            for (i in 0 until 30) { now += 40; t.offerForTest(frame(now, 4f)) }
            val duringFirst = t.autoFrameCount()
            assertTrue("first attempt should have buffered frames", duringFirst > 0)

            t.armAuto(now)
            assertEquals("re-arming starts from nothing", 0, t.autoFrameCount())
        } finally {
            t.close()
        }
    }

    /** Disarming drops the attempt without producing a result. */
    @Test
    fun disarmProducesNoOutcome() {
        val t = tracker()
        try {
            t.armAuto(0L)
            var now = 0L
            for (i in 0 until 30) { now += 40; t.offerForTest(frame(now, 4f)) }
            t.disarmAuto()
            assertNull("a disarmed attempt must not classify", t.takeAutoOutcome())
            assertEquals(AutoSegmenter.State.IDLE, t.autoState())
        } finally {
            t.close()
        }
    }

    /**
     * Arming then cancelling returns to idle and leaves nothing buffered.
     *
     * Regression: the armed state's button was originally a no-op, so the only
     * way out of "Ready" was to leave the tab. A user who tapped Start by
     * mistake was stuck looking at a screen that would never do anything.
     */
    @Test
    fun armThenCancelReturnsToIdle() {
        val t = tracker()
        try {
            t.armAuto(0L)
            assertEquals(AutoSegmenter.State.ARMED, t.autoState())
            t.disarmAuto()
            assertEquals(AutoSegmenter.State.IDLE, t.autoState())
            assertEquals("nothing may remain buffered", 0, t.autoFrameCount())
            assertNull("a cancelled attempt must not classify", t.takeAutoOutcome())
        } finally {
            t.close()
        }
    }

    /**
     * Switching camera mid-attempt must not splice two cameras' frames.
     *
     * The tracker is unbound and rebound on a switch. Disarming first means the
     * new camera starts from a clean buffer rather than continuing an attempt
     * whose first half came from the other lens.
     */
    @Test
    fun cameraSwitchDiscardsTheAttemptRatherThanSplicing() {
        val t = tracker()
        try {
            t.armAuto(0L)
            var now = 0L
            for (i in 0 until 30) { now += 40; t.offerForTest(frame(now, 4f)) }
            assertTrue(t.autoFrameCount() > 0)

            t.clear()
            assertEquals(AutoSegmenter.State.IDLE, t.autoState())
            assertEquals("no frames may survive the switch", 0, t.autoFrameCount())
            assertNull("the old attempt must not be classified", t.takeAutoOutcome())
        } finally {
            t.close()
        }
    }

    /**
     * Backgrounding must not leave an attempt armed over stale frames.
     *
     * Same lifecycle call the tab-switch path makes. On resume the user arms
     * again deliberately; nothing from before is reused.
     */
    @Test
    fun backgroundingSuspendsTheAttemptSafely() {
        val t = tracker()
        try {
            t.armAuto(0L)
            var now = 0L
            for (i in 0 until 40) { now += 40; t.offerForTest(frame(now, 5f)) }
            t.disarmAuto()          // the app goes to background

            // A long gap, then a frame from "after" the resume.
            t.offerForTest(frame(now + 30_000, 5f))
            assertNull("no inference may run on frames held across the gap",
                t.takeAutoOutcome())
            assertEquals(0, t.autoFrameCount())
        } finally {
            t.close()
        }
    }

    /**
     * Rapid repeated taps keep a single attempt.
     *
     * The armed state is entered again rather than a second attempt being
     * started alongside the first, so no duplicate outcome can be produced.
     */
    @Test
    fun rapidRepeatedStartsKeepOneAttempt() {
        val t = tracker()
        try {
            t.armAuto(0L)
            t.armAuto(10L)
            t.armAuto(20L)
            assertEquals(AutoSegmenter.State.ARMED, t.autoState())

            var now = 20L
            for (i in 0 until 12) { now += 40; t.offerForTest(frame(now, 0f)) }
            for (i in 0 until 40) {
                now += 40; t.offerForTest(frame(now, if (i % 2 == 0) 2f else 6f))
            }
            for (i in 0 until 60) { now += 40; t.offerForTest(frame(now, 0f)) }

            assertNotNull("exactly one outcome", t.takeAutoOutcome())
            assertNull("and not more than one", t.takeAutoOutcome())
        } finally {
            t.close()
        }
    }

    /** The manual fallback still works while a sign is running. */
    @Test
    fun manualFinishWorksDuringASign() {
        val t = tracker()
        try {
            t.armAuto(0L)
            var now = 0L
            for (i in 0 until 12) { now += 40; t.offerForTest(frame(now, 0f)) }
            for (i in 0 until 30) {
                now += 40; t.offerForTest(frame(now, if (i % 2 == 0) 2f else 6f))
            }
            val o = t.finishAutoManually(now)
            assertNotNull("Finish must end the attempt", o)
            assertTrue("the captured frames must be kept", (o?.frameCount ?: 0) > 0)
            assertNull("Finish already consumed the attempt", t.takeAutoOutcome())
            assertNull("a second Finish must not return another outcome", t.finishAutoManually(now))
        } finally {
            t.close()
        }
    }
}
