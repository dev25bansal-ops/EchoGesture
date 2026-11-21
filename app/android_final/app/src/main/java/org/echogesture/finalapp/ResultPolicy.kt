package org.echogesture.finalapp

/** Turns an ensemble ranking into a caption only when it meets both evidence floors. */
object ResultPolicy {
    const val MIN_MARGIN = 0.15f
    const val MIN_CLIP_SCORE_PERCENT = 65f
    const val MIN_CAMERA_SCORE_PERCENT = 85f

    fun isAccepted(margin: Float, scorePercent: Float, fromClip: Boolean = false): Boolean {
        val minimumScore = if (fromClip) MIN_CLIP_SCORE_PERCENT else MIN_CAMERA_SCORE_PERCENT
        return margin in MIN_MARGIN..1f && scorePercent in minimumScore..100f
    }
}
