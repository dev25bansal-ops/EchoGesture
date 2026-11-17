package org.echogesture.finalapp

/**
 * Decides whether a prediction is reliable enough to show as a recognised sign.
 *
 * Why this is separate from the classifier: the ensemble always returns a
 * distribution over 100 glosses, so it always has an argmax. On a real phone
 * recording it returned `FLIP` at 10.6% and `TAIL1` at 24.2% for input that
 * carried no readable sign. Presenting either as "the recognised sign" is a
 * false claim — the model has no notion of "none of these".
 *
 * The rule is deliberately conservative and stated in one place so it can be
 * measured. Two quantities decide it:
 *
 *  - **margin** = top-1 probability minus top-2. How *distinct* the winner is.
 *    Measured on 82 authorised validation clips: mean 0.378 when the top-1 was
 *    correct, 0.064 when it was wrong. This is the primary signal.
 *  - **score** = the top-1 probability. Used only as a floor against a flat,
 *    degenerate distribution — not as a second filter.
 *
 * Both are ranking quantities, never probabilities of correctness. The app must
 * not claim "94% likely correct"; it can say "best match".
 *
 * Measured coverage on that population: a 0.15 margin keeps 61% of captures at
 * 94% accuracy. This is a real trade — 39% of attempts are declined rather than
 * guessed, which is the correct direction for an accessibility tool.
 */
object ResultPolicy {

    /** Minimum top-1 minus top-2 probability for a reading to be shown. */
    const val MIN_MARGIN = 0.15f

    /**
     * Minimum top-1 probability. A guard against a pathological flat output
     * rather than a second filter: on the measured population no capture with
     * a 0.15 margin scored below 0.20, so this never changed the accepted set.
     */
    const val MIN_SCORE_PERCENT = 20f

    sealed interface Acceptance {
        data object Accepted : Acceptance
        data class Declined(val reason: Reason) : Acceptance
    }

    enum class Reason {
        /** The winner is barely ahead of the runner-up. */
        TOO_CLOSE,
        /** The whole distribution is flat. */
        LOW_SCORE,
    }

    /**
     * @param margin top-1 minus top-2 probability, in [0, 1].
     * @param scorePercent top-1 probability as a percentage, in [0, 100].
     */
    fun evaluate(margin: Float, scorePercent: Float): Acceptance = when {
        margin < MIN_MARGIN -> Acceptance.Declined(Reason.TOO_CLOSE)
        scorePercent < MIN_SCORE_PERCENT -> Acceptance.Declined(Reason.LOW_SCORE)
        else -> Acceptance.Accepted
    }

    /** True when the reading may be spoken aloud or presented as a gloss. */
    fun isAccepted(margin: Float, scorePercent: Float): Boolean =
        evaluate(margin, scorePercent) is Acceptance.Accepted
}
