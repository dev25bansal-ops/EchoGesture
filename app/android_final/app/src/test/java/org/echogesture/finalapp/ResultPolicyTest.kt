package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the boundary the HUD's accepted/declined wording depends on.
 *
 * `CameraScreen` paints the live pill from `ResultPolicy.isAccepted`: a reading
 * that passes gets "Recognised sign" in success green, one that is declined gets
 * "Uncertain reading" in warning amber. That is an accessibility claim, so the
 * two states must never blur. Two failure modes are checked here:
 *
 *  - a reading that is declined must not report as accepted (a green pill over a
 *    reading the model is unsure about is the exact false confidence the gate
 *    exists to prevent);
 *  - the boundary itself must be closed and consistent, so the pill and the
 *    caption cannot disagree at the equality point.
 */
class ResultPolicyTest {

    // A comfortably accepted reading: distinct winner, high score.
    private val clearMargin = 0.60f
    private val clearScore = 88f

    @Test
    fun aDistinctHighScoringReadingIsAccepted() {
        assertTrue(ResultPolicy.isAccepted(clearMargin, clearScore))
        assertEquals(ResultPolicy.Acceptance.Accepted,
            ResultPolicy.evaluate(clearMargin, clearScore))
    }

    @Test
    fun aCloseCallIsDeclined() {
        // The runner-up is nearly tied with the winner.
        assertFalse(ResultPolicy.isAccepted(ResultPolicy.MIN_MARGIN - 0.01f, clearScore))
        assertEquals(ResultPolicy.Acceptance.Declined(ResultPolicy.Reason.TOO_CLOSE),
            ResultPolicy.evaluate(ResultPolicy.MIN_MARGIN - 0.01f, clearScore))
    }

    @Test
    fun aFlatDistributionIsDeclinedOnScore() {
        // A wide margin between two low probabilities still fails the floor.
        assertFalse(ResultPolicy.isAccepted(clearMargin, ResultPolicy.MIN_SCORE_PERCENT - 0.01f))
        assertEquals(ResultPolicy.Acceptance.Declined(ResultPolicy.Reason.LOW_SCORE),
            ResultPolicy.evaluate(clearMargin, ResultPolicy.MIN_SCORE_PERCENT - 0.01f))
    }

    @Test
    fun theAcceptedBoundaryIsClosedAtBothThresholds() {
        // At exactly the thresholds the reading is accepted...
        assertTrue("margin == MIN_MARGIN must be accepted",
            ResultPolicy.isAccepted(ResultPolicy.MIN_MARGIN, clearScore))
        assertTrue("score == MIN_SCORE_PERCENT must be accepted",
            ResultPolicy.isAccepted(clearMargin, ResultPolicy.MIN_SCORE_PERCENT))
        // ...and one ULP below on either axis it must not be.
        assertFalse(ResultPolicy.isAccepted(
            Math.nextDown(ResultPolicy.MIN_MARGIN), clearScore))
        assertFalse(ResultPolicy.isAccepted(
            clearMargin, Math.nextDown(ResultPolicy.MIN_SCORE_PERCENT)))
    }

    @Test
    fun isAcceptedAgreesWithEvaluateOnEveryDeclineReason() {
        // The pill reads isAccepted; the caption reads the decline reason. They
        // must classify identically or the two halves of the result screen can
        // contradict each other.
        val cases = listOf(
            clearMargin to clearScore,
            0f to 0f,
            ResultPolicy.MIN_MARGIN to 0f,
            0f to 100f,
            ResultPolicy.MIN_MARGIN - 0.01f to ResultPolicy.MIN_SCORE_PERCENT - 0.01f,
        )
        for ((margin, score) in cases) {
            val accepted = ResultPolicy.evaluate(margin, score) is ResultPolicy.Acceptance.Accepted
            assertEquals("isAccepted disagreed with evaluate at margin=$margin score=$score",
                accepted, ResultPolicy.isAccepted(margin, score))
        }
    }
}