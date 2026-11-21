package org.echogesture.finalapp

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ResultPolicyTest {
    @Test
    fun clipsDeclineThePreviouslyAcceptedWrongReading() {
        assertFalse(ResultPolicy.isAccepted(0.544f, 62.012f, fromClip = true))
        assertTrue(ResultPolicy.isAccepted(0.15f, 65f, fromClip = true))
        assertFalse(ResultPolicy.isAccepted(0.149f, 99f, fromClip = true))
    }

    @Test
    fun cameraDeclinesTheRecordedRestingHandScores() {
        assertFalse(ResultPolicy.isAccepted(0.728f, 80.2f))
        assertTrue(ResultPolicy.isAccepted(0.15f, 85f))
    }

    @Test
    fun invalidScoresNeverBecomeCaptions() {
        assertFalse(ResultPolicy.isAccepted(Float.NaN, 95f))
        assertFalse(ResultPolicy.isAccepted(0.8f, Float.POSITIVE_INFINITY))
        assertFalse(ResultPolicy.isAccepted(1.01f, 95f))
    }
}
