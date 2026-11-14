package org.echogesture.finalapp

/**
 * Picks the part of a recording that actually contains the sign.
 *
 * Why this exists: the signer brackets the sign with their own presses, and a
 * real press is rarely frame-accurate. Holding Start a second early, or
 * pressing Finish a beat late, wraps the sign in idle frames. Those frames are
 * then spread across the 32 samples the model gets, so the sign occupies fewer
 * of the 24 temporal steps — and because `features.py` computes velocity as a
 * plain per-frame difference with no `dt`, the motion features are scaled down
 * by exactly that dilution.
 *
 * Measured on 40 validation clips with 2 s of idle before the sign:
 *
 *   whole recording      20/40 top-1
 *   trimmed to the sign  29/40 top-1   (clean recording is 31/40)
 *
 * and it improved clean recordings too: 31/32 on train with no idle, up from
 * 29/32 untrimmed, so trimming does not merely undo padding.
 *
 * What this deliberately is NOT: a confidence search. Choosing the crop the
 * model likes best would select the answer by how much the model agrees with
 * itself, which is not evidence. The trim uses only hand presence and landmark
 * movement — quantities the phone measures directly.
 *
 * It is also conservative by construction. It only ever removes a leading or
 * trailing run; it never reorders, never drops the middle of a sign, and never
 * pads. If nothing moves, it returns the whole recording rather than inventing
 * a boundary.
 */
object SignSegmenter {

    /** What the trim decided, for diagnostics and the debug timeline. */
    data class Segment(
        val startIndex: Int,
        val endIndex: Int,
        val reason: Reason
    ) {
        /** True when the whole recording is used. */
        val isWhole: Boolean get() = startIndex == 0 && reason == Reason.NO_MOTION
    }

    enum class Reason {
        /** Movement found; the recording is trimmed to it. */
        TRIMMED,
        /** Nothing moved enough to trim against; the whole recording is used. */
        NO_MOTION,
        /** Fewer frames than the minimum needed to judge; used as recorded. */
        TOO_FEW_FRAMES,
        /** Hands never appeared; left for the coverage gate to reject. */
        NO_HANDS,
    }

    /**
     * Movement below this fraction of the recording's own peak is treated as
     * idle. Relative rather than absolute so a slow signer is not trimmed to
     * nothing and a fast one is not clipped.
     */
    const val MOTION_THRESHOLD_FRACTION = 0.25f

    /**
     * Padding kept either side of the moving part, in seconds. A sign has
     * stationary moments at its onset and offset — the hand holds a shape
     * before moving — so trimming exactly to the first moving frame would cut
     * the very frames that carry the initial handshape.
     */
    const val PAD_SECONDS = 0.5f

    /** Below this many frames the movement estimate is noise; keep everything. */
    const val MIN_FRAMES = 8

    /**
     * Mean absolute landmark displacement between consecutive frames, per
     * frame. Zero on frames where either side has no hand, because a hand
     * leaving or entering is not evidence of the sign moving.
     */
    fun displacement(frames: List<HandTracker.Frame>): FloatArray {
        val out = FloatArray(frames.size)
        for (i in 1 until frames.size) {
            val prev = frames[i - 1]
            val cur = frames[i]
            if (prev.hands.isEmpty() || cur.hands.isEmpty()) continue
            var sum = 0.0
            var count = 0
            // Compare per slot so a hand swapping sides is not read as motion.
            for (hand in cur.hands) {
                val before = prev.hands.firstOrNull { it.slot == hand.slot } ?: continue
                for (j in 0 until minOf(before.xyz.size, hand.xyz.size)) {
                    sum += kotlin.math.abs(hand.xyz[j] - before.xyz[j])
                    count++
                }
            }
            if (count > 0) out[i] = (sum / count).toFloat()
        }
        return out
    }

    /**
     * Chooses the sub-interval to hand the model.
     *
     * @param fps the recording's achieved frame rate, used to size the padding
     *            in frames. Padding is a real duration, not a frame count, so
     *            a slow recording still keeps half a second either side.
     */
    fun choose(
        frames: List<HandTracker.Frame>,
        fps: Float,
        thresholdFraction: Float = MOTION_THRESHOLD_FRACTION,
        padSeconds: Float = PAD_SECONDS
    ): Segment {
        val n = frames.size
        if (n < MIN_FRAMES) return Segment(0, n, Reason.TOO_FEW_FRAMES)
        if (frames.none { it.hands.isNotEmpty() }) return Segment(0, n, Reason.NO_HANDS)

        val d = displacement(frames)
        val peak = d.max()
        if (peak <= 0f) return Segment(0, n, Reason.NO_MOTION)

        val threshold = peak * thresholdFraction
        var first = -1
        var last = -1
        for (i in d.indices) {
            if (d[i] >= threshold) {
                if (first < 0) first = i
                last = i
            }
        }
        if (first < 0 || last <= first) return Segment(0, n, Reason.NO_MOTION)

        val rate = if (fps.isFinite() && fps > 1f) fps else 30f
        val pad = (padSeconds * rate).toInt().coerceAtLeast(1)
        val start = (first - pad).coerceAtLeast(0)
        val end = (last + pad + 1).coerceAtMost(n)
        return Segment(start, end, Reason.TRIMMED)
    }
}