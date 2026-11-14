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
 *   trimmed to the sign  29/40 top-1
 *
 * Trimming pays when there is idle to remove, and it is uncertain when there
 * is not. Measured on the same 32 train and 40 validation clips, with idle
 * added as frames carrying no hand:
 *
 *                        train        validation
 *   clean    whole       29/32        31/40
 *            trimmed     31/32        29/40     <- opposite signs
 *   2 s lead whole       24/32        20/40
 *            trimmed     31/32        29/40     <- large gain, both splits
 *
 * So the clean-recording effect flips direction between splits (n=32, n=40;
 * a 3-clip swing is inside the noise), while the benefit on a recording with
 * real idle is large and consistent. Trimming is therefore worth doing, but
 * "never hurts a clean recording" is NOT a claim this data supports and must
 * not be made. An earlier version of this comment did make it, by citing
 * train's 31/32 without validation's 29/40.
 *
 * Two alternatives were measured and rejected (aggregate over both splits and
 * three idle conditions, 216 evaluations, unsmoothed whole-interval basis):
 *
 *   whole interval, no trim                    131/216
 *   unconditional motion trim                  180/216
 *   conservative gated trim (long flat idle)   146/216
 *   motion trim with a "keep >=45%" floor      133/216
 *
 * The gate was rejected because clean clips have leading near-zero runs up to
 * 4.07 s, which overlaps the idle it is meant to detect: it either fires on
 * real signs or stops catching real idle. The floor was rejected because it
 * disables exactly the trims that help — the ones that remove most of the
 * padding. The unconditional trim is kept because its downside is bounded and
 * small while its upside is large and repeatable.
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
        /** True when this reason leaves the recording as it was captured. */
        val isWhole: Boolean get() = reason != Reason.TRIMMED
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
        /** The motion trim would have kept almost nothing; the whole recording is used. */
        TRIMMED_TOO_LITTLE,
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
     * A trim keeping less than this fraction of the recording is treated as a
     * misfire and the whole recording is used instead. Far below any real trim,
     * so it guards against a blip without disabling genuine trimming.
     */
    const val DEGENERATE_KEEP_FRACTION = 0.10f

    /**
     * Percentile of the moving-frame displacements used as the motion
     * reference. Above the median so a slow sign is not judged by its average
     * frame, below the maximum so a spike cannot set it.
     */
    const val PEAK_PERCENTILE = 95

    /**
     * A high percentile of the non-zero displacements, or 0 if nothing moved.
     *
     * Same device-hazard reasoning as [DEGENERATE_KEEP_FRACTION]: the phone
     * measures displacement directly, and a single noisy frame is not evidence
     * about how much the sign moved.
     */
    fun robustPeak(displacement: FloatArray): Float {
        val moved = displacement.filter { it > 0f }.sorted()
        if (moved.isEmpty()) return 0f
        val idx = ((moved.size - 1) * PEAK_PERCENTILE / 100).coerceIn(0, moved.size - 1)
        return moved[idx]
    }

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
        // Reference is a high percentile of the MOVING frames, not the raw
        // maximum. A single jitter spike otherwise raises the threshold above
        // every genuine moving frame and the window collapses onto the spike —
        // a demonstrated defect: one spiked frame cut a 91-frame window to 32.
        // The percentile is unaffected by a handful of outliers.
        val peak = robustPeak(d)
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
        var start = (first - pad).coerceAtLeast(0)
        var end = (last + pad + 1).coerceAtMost(n)

        // A trim that keeps almost nothing is a misfire, not a trim: a brief
        // detector blip in a long recording would otherwise become the whole
        // "sign". Keeping the whole recording is the safe answer. The floor is
        // deliberately far below any real trim (measured median kept fraction
        // is 0.64 clean and 0.27 heavily padded) so it never disables a genuine
        // trim; a floor high enough to prevent clean-clip regressions would
        // also disable the trims that remove real padding, and was measured to
        // cost the entire benefit.
        if (end - start < (n * DEGENERATE_KEEP_FRACTION).toInt().coerceAtLeast(1)) {
            start = 0
            end = n
            return Segment(start, end, Reason.TRIMMED_TOO_LITTLE)
        }
        return Segment(start, end, Reason.TRIMMED)
    }
}