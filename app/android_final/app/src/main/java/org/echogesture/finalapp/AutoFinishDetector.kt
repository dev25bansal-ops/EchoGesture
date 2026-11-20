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
    private var quietSinceMs = Long.MIN_VALUE

    /** Moving frames seen so far, i.e. how much actual motion this holds. */
    private var motionFrames = 0

    /**
     * Every displacement seen so far this recording.
     *
     * A running maximum was the first implementation and it is WRONG for the
     * same reason SignSegmenter.robustPeak exists: one glitch frame permanently
     * raises the bar. Measured failure — a one-frame handedness flip makes
     * SignSegmenter.displacement pair the wrong hands and spikes to ~0.28 while
     * the real sign moves at ~0.03, so the quiet threshold became 0.042 and
     * every genuine sign frame counted as "quiet": the sign was auto-finished
     * about 1 s in, half-performed.
     *
     * Keeping the history lets the peak be a high percentile instead, which a
     * handful of outliers cannot move.
     */
    private val history = ArrayList<Float>(256)

    /** True once real motion has been seen; exposed for the live status line. */
    val hasStartedMotion: Boolean get() = started

    /** Internal state, for tests and debugging a verdict that never fires. */
    fun debugState(): String =
        "started=$started motionFrames=$motionFrames peak=${"%.4f".format(peak())} " +
            "quietSince=$quietSinceMs quietMs=${quietSinceMs} recordedCap=$minRecordingMs" 

    /** High percentile of the motion seen so far; 0 before anything moves. */
    private fun peak(): Float {
        if (history.isEmpty()) return 0f
        val moved = history.filter { it > 0f }.sorted()
        if (moved.isEmpty()) return 0f
        val idx = ((moved.size - 1) * PEAK_PERCENTILE / 100).coerceIn(0, moved.size - 1)
        return moved[idx]
    }

    /**
     * Feeds one frame's [displacement] (as computed by
     * [SignSegmenter.displacement]) at wall-clock [nowMs].
     *
     * @param recordedMs how long this recording has been running.
     */
    fun onFrame(displacement: Float, nowMs: Long, recordedMs: Long): Verdict {
        if (recordedMs >= maxRecordingMs) return Verdict.TIMED_OUT

        // The peak is computed BEFORE this frame joins the history, so the
        // phase-1 gate compares against the motion already seen rather than
        // against itself. With the old ordering, `displacement >= peak *
        // startFraction` reduced to `displacement >= 0.4 * displacement` — a
        // test that is true for ANY positive value. START_FRACTION could never
        // reject, so merely raising the hand started the sign.
        val priorPeak = peak()
        history.add(displacement)
        if (history.size > HISTORY_CAP) history.removeAt(0)

        // Phase 1: wait for the sign to begin.
        if (!started) {
            // Two pieces of evidence: this frame moves meaningfully on its own
            // (an absolute floor, so sensor noise cannot start a sign), AND it
            // is comparable to the motion already seen (relative, so a slow
            // signer and a fast one behave alike).
            val aboveFloor = displacement >= NOISE_FLOOR
            val abovePrior = priorPeak <= 0f || displacement >= priorPeak * startFraction
            if (aboveFloor && abovePrior) {
                started = true
                quietSinceMs = Long.MIN_VALUE
            }
            return Verdict.CONTINUE
        }

        // Phase 2: has it settled? Judged against the percentile peak, so a
        // single spike cannot raise the bar for the rest of the recording.
        val peakNow = peak()
        val quiet = displacement <= peakNow * stopFraction
        // Count the moving frame BEFORE branching, not inside the !quiet branch.
        //
        // Counting it inside meant a frame that moved but sat below the rising
        // threshold was neither "quiet" nor "motion", so a short clip could
        // never accumulate MIN_MOTION_FRAMES. Measured on the CHEW clip: 23
        // frames over 726 ms, motion seen on almost all of them, and the
        // detector still reported motionFrames < 6 and never fired.
        if (!quiet) motionFrames++
        if (!quiet) {
            quietSinceMs = Long.MIN_VALUE
            return Verdict.CONTINUE
        }
        if (quietSinceMs == Long.MIN_VALUE) {
            quietSinceMs = nowMs
            return Verdict.CONTINUE
        }
        // Quiet time accumulated BEFORE the minimum does not count -- but do
        // NOT reset the clock on every tick.
        //
        // The previous version did `quietSinceMs = nowMs` here, which meant that
        // while recordedMs stayed under the minimum the quiet window restarted
        // continuously and could never grow. Measured on the CHEW clip:
        // quietSince=1188 against a 1200 ms minimum, forever, so a complete
        // 726 ms sign followed by a second of stillness never finished.
        //
        // Holding quietSinceMs at MIN_VALUE instead makes the window START at
        // the minimum, which is the intended meaning: quiet before the minimum
        // is not counted, quiet after it is.
        if (recordedMs < minRecordingMs) {
            quietSinceMs = Long.MIN_VALUE
            return Verdict.CONTINUE
        }
        // First tick at or past the minimum: start the window here.
        if (quietSinceMs == Long.MIN_VALUE) {
            quietSinceMs = nowMs
            return Verdict.CONTINUE
        }
        // A sign needs SUSTAINED motion, not just the hand arriving.
        //
        // This is the second half of the "raise the hand, pause to recall the
        // sign" bug. The hand's arrival clears the noise floor and legitimately
        // starts the detector, so magnitude alone cannot tell arrival from
        // signing. What separates them is that arrival is one or two frames and
        // a sign is many: measured p25 of real clip motion is 0.15 per frame
        // sustained across the clip. Requiring several moving frames means an
        // arrival-plus-pause records nothing, and the user simply keeps going.
        if (motionFrames < MIN_MOTION_FRAMES) {
            quietSinceMs = nowMs
            return Verdict.CONTINUE
        }
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
         * Smallest per-frame motion that can start a sign, in the units
         * SignSegmenter.displacement returns (mean absolute normalised landmark
         * movement between consecutive frames).
         *
         * Without an absolute floor the start gate is satisfiable by sensor
         * noise: a still hand still jitters, and that jitter would begin a sign
         * and then immediately look settled. 0.01 is below the smallest genuine
         * motion measured on the corpus (p25 of real clips is 0.15) and above
         * the jitter of a held hand.
         */
        const val NOISE_FLOOR = 0.01f

        /**
         * Moving frames a recording must accumulate before it can auto-finish.
         *
         * Separates "the hand arrived" from "a sign was performed": an arrival
         * is one or two frames, a sign is many. Without this, raising the hand
         * and pausing to recall the sign ends the recording before it begins.
         * Sizing this is a genuine trade-off, and the two failures are opposite:
         *
         *  - Too HIGH and a short sign never auto-finishes. 6 required ~0.6 s of
         *    motion at the 10 fps the analyser achieves; the shortest clip
         *    measured here is 726 ms with 23 frames, and most of its frames sit
         *    below the rising quiet threshold, so 6 was unreachable and the
         *    detector never fired.
         *  - Too LOW and "raise the hand, pause to think" finishes the recording
         *    before the signer starts. At 4 the unit test
         *    raisingTheHandAndPausingToThinkDoesNotFinish started failing
         *    again, which is why this is back at 6.
         *
         * The real fix for the first case was counting motion correctly (see
         * onFrame), not moving this number. With that corrected, 6 is reachable
         * on a 726 ms clip -- the moving frames are now counted rather than
         * silently skipped.
         */
        const val MIN_MOTION_FRAMES = 6

        /** Percentile of displacements used as the motion reference. */
        const val PEAK_PERCENTILE = 95

        /** Displacements retained for the percentile; ~8 s at 30 fps. */
        const val HISTORY_CAP = 256

        /**
         * Fraction of peak motion below which the hand counts as still.
         * Matches SignSegmenter.PAD logic in spirit: the endpoints of a sign
         * are stationary, so this has to sit well below the moving frames.
         */
        const val STOP_FRACTION = 0.15f
    }
}
