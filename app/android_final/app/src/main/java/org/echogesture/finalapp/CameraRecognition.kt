package org.echogesture.finalapp

internal data class ScoredRecording(
    val result: LandmarkClassifier.Result,
    val frames: List<HandTracker.Frame>,
    val mask: FloatArray,
    val corroborated: Boolean = false
)

// At moderate scores, require all nine models and a clear lead over the runner-up.
private fun supportsCameraMatch(result: LandmarkClassifier.Result): Boolean =
    result.confidence >= 0.40f && result.margin >= 0.35f && result.agreeingMembers == 9

internal fun scoreRecording(
    classifier: LandmarkClassifier,
    recording: SignRecorder.Recording,
    samples: List<HandTracker.Frame>,
    autoCapture: Boolean
): ScoredRecording {
    fun score(frames: List<HandTracker.Frame>): ScoredRecording {
        val tensors = RecordingFeatures.buildBoth(frames)
        return ScoredRecording(classifier.predict(tensors.featuresXY, tensors.mask, tensors.featuresXYZ),
            frames, tensors.mask)
    }
    val primary = score(samples)
    if (!autoCapture || !ResultPolicy.hasSustainedTracking(samples)) return primary
    val first = primary.result
    if (ResultPolicy.isAccepted(first.margin, first.confidence * 100f)) return primary
    if (first.confidence < 0.65f && !supportsCameraMatch(first)) return primary

    val visible = RecordingFeatures.visibleSamples(recording)
    if (visible == samples || !ResultPolicy.hasSustainedTracking(visible)) return primary
    val retry = score(visible)
    val confirmed = primary.result.confidence >= 0.65f && retry.result.prediction == primary.result.prediction &&
        ResultPolicy.isAccepted(retry.result.margin, retry.result.confidence * 100f)
    val corroborated = retry.result.prediction == primary.result.prediction &&
        supportsCameraMatch(primary.result) && supportsCameraMatch(retry.result) &&
        SignSegmenter.displacement(visible).count { it > 0.008f } >= 3
    val chosen = if (confirmed) retry else primary
    // Both passes contribute to the inference time even if the first result is retained.
    val timings = primary.result.memberMillis.mapValues { (name, time) ->
        time + retry.result.memberMillis.getValue(name)
    }
    return chosen.copy(corroborated = corroborated, result = chosen.result.copy(
        inferenceMillis = primary.result.inferenceMillis + retry.result.inferenceMillis,
        memberMillis = timings
    ))
}

/** A framing hint only. Measured in frame widths, so a tall portrait preview uses the same rule. */
internal fun handsLookDistant(frames: List<HandTracker.Frame>): Boolean {
    var tracked = 0
    var small = 0
    for (frame in frames) {
        if (frame.hands.isEmpty()) continue
        tracked++
        val allSmall = frame.hands.all { hand ->
            var minX = Float.POSITIVE_INFINITY
            var minY = Float.POSITIVE_INFINITY
            var maxX = Float.NEGATIVE_INFINITY
            var maxY = Float.NEGATIVE_INFINITY
            for (i in hand.xyz.indices step 3) {
                val x = hand.xyz[i]
                val y = hand.xyz[i + 1]
                minX = minOf(minX, x)
                maxX = maxOf(maxX, x)
                minY = minOf(minY, y)
                maxY = maxOf(maxY, y)
            }
            // Live y coordinates include the classifier's additional 4:3 scale.
            maxOf(maxX - minX, (maxY - minY) * 0.75f) < 0.10f
        }
        if (allSmall) small++
    }
    return tracked >= 4 && small * 2 > tracked
}
