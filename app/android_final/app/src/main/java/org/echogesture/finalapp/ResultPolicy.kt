package org.echogesture.finalapp

/** Turns an ensemble ranking into a caption only when it meets both evidence floors. */
object ResultPolicy {
    const val MIN_MARGIN = 0.15f
    const val MIN_CLIP_SCORE_PERCENT = 65f
    const val MIN_CAMERA_SCORE_PERCENT = 85f

    /** Brief detector flashes can produce confident guesses from just a few handshapes. */
    fun hasSustainedTracking(frames: List<HandTracker.Frame>): Boolean {
        var previous: Long? = null
        var observedMs = 0L
        var observations = 0
        for (frame in frames) {
            if (frame.hands.isEmpty()) {
                previous = null
                continue
            }
            observations++
            previous?.let { observedMs += (frame.capturedAtMs - it).coerceIn(0L, 150L) }
            if (observations >= 4 && observedMs >= 250L) return true
            previous = frame.capturedAtMs
        }
        return false
    }

    fun isAccepted(margin: Float, scorePercent: Float, fromClip: Boolean = false): Boolean {
        val minimumScore = if (fromClip) MIN_CLIP_SCORE_PERCENT else MIN_CAMERA_SCORE_PERCENT
        return margin in MIN_MARGIN..1f && scorePercent in minimumScore..100f
    }
}
