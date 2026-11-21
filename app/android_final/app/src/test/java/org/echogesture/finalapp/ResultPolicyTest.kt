package org.echogesture.finalapp

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ResultPolicyTest {
    @Test
    fun isolatedHandDetectionsCannotConfirmACameraCaption() {
        val hand = HandTracker.Hand(0, FloatArray(63))
        fun frame(ms: Long) = HandTracker.Frame(listOf(hand), ms)
        val gap = HandTracker.Frame(emptyList(), 120)
        assertFalse(ResultPolicy.hasSustainedTracking(listOf(frame(0), frame(70), gap, frame(500), frame(580))))
        assertFalse(ResultPolicy.hasSustainedTracking(listOf(frame(0), frame(900))))
        assertTrue(ResultPolicy.hasSustainedTracking(listOf(frame(0), frame(90), frame(180), frame(270))))
    }

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
