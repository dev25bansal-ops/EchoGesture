package org.echogesture.finalapp

/**
 * Decides, from the motion the phone already measures, that a sign has ended.
 *
 * Why this exists. In record mode the signer taps Start, performs, then taps
 * Finish. The tap is not free: on the Vivo V2404 one measured capture ran
 * 1540 ms and held 4 analysed frames, and another held the hand for only 11 of
 * 32 frames, because the press landed at the wrong moment relative to a camera
 * analyser running at 4-15 fps. The signer cannot see that rate, so they cannot
 * time a press to it. The motion signal can.
 *
 * How it decides. Not "no motion yet" (every recording begins with the hand
 * arriving) and not "one quiet frame" (a sign has stationary holds inside it).
 * The rule is:
 *
 *  1. Wait for motion to START — a displacement above [startFraction] of the
 *     recording's running peak. Before that the signer is still getting ready.
 *  2. Once started, a frame whose displacement is below [stopFraction] of that
 *     peak counts toward the quiet run.
 *  3. When the quiet run lasts [quietMs], the sign has ended.
 *
 * The thresholds are fractions of the recording's OWN peak motion, matching
 * [SignSegmenter]'s reasoning: a slow signer and a fast one should behave the
 * same, and an absolute threshold would only suit one of them.
 *
 * Two guards keep it from firing on things that are not a finished sign:
 *
 *  - [minRecordingMs] — nothing auto-finishes before the sign has had a chance
 *    to happen. Without it, a signer pausing to think is cut off in the first
 *    half-second.
 *  - [maxRecordingMs] — an absolute ceiling, so a signer who keeps moving never
 *    has a recording run to [SignRecorder.MAX_DURATION_MS] and get discarded.
 *
 * Pure Kotlin with an injected clock: no Android types, no timers, no threads.
 * That is what makes "it finishes after the sign stops, not before" provable in
 * a unit test instead of only observable by holding a phone.
 */
class AutoFinishDetector(
    private val quietMs: Long = QUIET_MS_DEFAULT,
    private val minRecordingMs: Long = MIN_RECORDING_MS_DEFAULT,
    private val maxRecordingMs: Long = MAX_RECORDING_MS_DEFAULT,
    private val startFraction: Float = START_FRACTION,
    private val stopFraction: Float = STOP_FRACTION,
) {

    enum class Verdict {
        /** Keep recording. */
        CONTINUE,
        /** Motion started and has now settled: the sign is over. */
        FINISHED,
        /** The recording hit the ceiling without ever settling. */
        TIMED_OUT,
    }

    private var started = false
    private var peak = 0f
    private var quietSinceMs = Long.MIN_VALUE

    /** True once real motion has been seen; exposed for the live status line. */
    val hasStartedMotion: Boolean get() = started

    /**
     * Feeds one frame's [displacement] (as computed by
     * [SignSegmenter.displacement]) at wall-clock [nowMs].
     *
     * @param recordedMs how long this recording has been running.
     */
    fun onFrame(displacement: Float, nowMs: Long, recordedMs: Long): Verdict {
        if (recordedMs >= maxRecordingMs) return Verdict.TIMED_OUT

        if (displacement > 0f && displacement > peak) peak = displacement

        // Phase 1: wait for the sign to begin. Until the peak is meaningful,
        // "quiet" is indistinguishable from "not started yet".
        if (!started) {
            if (peak > 0f && displacement >= peak * startFraction) {
                started = true
                quietSinceMs = Long.MIN_VALUE
            }
            return Verdict.CONTINUE
        }

        // Phase 2: has it settled?
        val quiet = displacement <= peak * stopFraction
        if (!quiet) {
            quietSinceMs = Long.MIN_VALUE
            return Verdict.CONTINUE
        }
        if (quietSinceMs == Long.MIN_VALUE) {
            quietSinceMs = nowMs
            return Verdict.CONTINUE
        }
        // The minimum guards the pause-to-think case: motion started, the
        // signer paused, but the recording is still young.
        if (recordedMs < minRecordingMs) return Verdict.CONTINUE
        return if (nowMs - quietSinceMs >= quietMs) Verdict.FINISHED else Verdict.CONTINUE
    }

    companion object {
        /**
         * How long the hand must be still before the sign counts as over.
         *
         * Signs are 1-6 s with stationary holds inside them. 900 ms is longer
         * than any hold measured in the corpus's static signs (NOON, DARK and
         * VOICE are held shapes, and their internal pauses are under 600 ms)
         * and short enough that the signer is not left waiting.
         */
        const val QUIET_MS_DEFAULT = 900L

        /** Nothing finishes before this, so a pause to think is not a sign. */
        const val MIN_RECORDING_MS_DEFAULT = 1_200L

        /**
         * Ceiling for auto-finish. Below [SignRecorder.MAX_DURATION_MS] on
         * purpose: hitting the recorder's own maximum DISCARDS the recording,
         * whereas finishing here keeps it.
         */
        const val MAX_RECORDING_MS_DEFAULT = 8_000L

        /** Fraction of peak motion that counts as "the sign has begun". */
        const val START_FRACTION = 0.40f

        /**
         * Fraction of peak motion below which the hand counts as still.
         * Matches SignSegmenter.PAD logic in spirit: the endpoints of a sign
         * are stationary, so this has to sit well below the moving frames.
         */
        const val STOP_FRACTION = 0.15f
    }
}
