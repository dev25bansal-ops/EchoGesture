package org.echogesture.finalapp

/**
 * Turns one recorded interval into the model's 24×255 input.
 *
 * The training path is two stages and this reproduces both:
 *
 *  1. `read_frames()` samples 32 frames spread evenly across the WHOLE clip.
 *  2. `build_features(expected_frames=24)` resamples those 32 to 24.
 *
 * Skipping stage 1 and going straight from N recorded frames to 24 would look
 * equivalent — both are uniform in time — but it double-rounds differently and
 * silently changes the timestamp of every model step. The point of this mode
 * is that the model sees what it saw in training, so both stages are kept.
 *
 * Nothing is invented. Every one of the 32 points is a frame the camera really
 * analysed. If the recording is shorter than 32 frames the points are passed
 * through as they are and the caller resamples upstream, exactly as a short
 * training clip behaves.
 */
object RecordingFeatures {

    /** Frames sampled across the interval, matching `read_frames(max_frames=32)`. */
    const val SAMPLED = 32

    /** Temporal steps the model consumes. */
    const val MODEL_STEPS = 24

    /**
     * The frames the model will see: the sign-bearing part of the recording,
     * chosen evenly by real captured timestamp.
     *
     * The trim matters. Measured on 40 validation clips with 2 s of idle before
     * the sign, sampling the whole recording scored 20/40 against 29/40 for the
     * trimmed interval (a clean recording scores 31/40). See [SignSegmenter].
     */
    fun sample(recording: SignRecorder.Recording): List<HandTracker.Frame> =
        sample(recording, fpsOf(recording))

    fun sample(recording: SignRecorder.Recording, fps: Float): List<HandTracker.Frame> {
        val seg = SignSegmenter.choose(recording.frames, fps)
        val frames = recording.frames.subList(seg.startIndex, seg.endIndex)
        if (frames.size <= SAMPLED) return frames
        val idx = HandTracker.evenlySpacedIndices(
            LongArray(frames.size) { frames[it].capturedAtMs },
            SAMPLED
        )
        return idx.map { frames[it] }
    }

    /** The part of the recording that was kept, for diagnostics. */
    fun segment(recording: SignRecorder.Recording): SignSegmenter.Segment =
        SignSegmenter.choose(recording.frames, fpsOf(recording))

    /** Achieved frame rate of the recording, from its own timestamps. */
    fun fpsOf(recording: SignRecorder.Recording): Float {
        val n = recording.frameCount
        if (n < 2 || recording.durationMs <= 0) return 0f
        return (n - 1) * 1000f / recording.durationMs
    }

    /** Flatten sampled frames into the (T, 2, 21, 3) + (T, 2) contract. */
    fun tensors(frames: List<HandTracker.Frame>): Triple<FloatArray, FloatArray, Int> {
        val n = frames.size
        val landmarks = FloatArray(n * 2 * 21 * 3)
        val present = FloatArray(n * 2)
        frames.forEachIndexed { fi, fr ->
            for (hand in fr.hands) {
                present[fi * 2 + hand.slot] = 1f
                System.arraycopy(hand.xyz, 0, landmarks, (fi * 2 + hand.slot) * 21 * 3, 21 * 3)
            }
        }
        return Triple(landmarks, present, n)
    }

    /** The full 24×255 tensor for one recording. */
    fun build(recording: SignRecorder.Recording): FloatArray =
        buildBoth(recording).let { it.featuresXY ?: FloatArray(0) }

    /** Both feature widths plus the attention mask, from ONE landmark pass. */
    data class Tensors(
        val featuresXY: FloatArray,
        val featuresXYZ: FloatArray,
        val mask: FloatArray
    )

    /**
     * Builds the 255-wide and 381-wide tensors and the mask together.
     *
     * The two widths are the same landmarks read with and without z, so the
     * mask is identical for both and MediaPipe is never run twice. Building
     * them separately would risk the mask drifting from the features it is
     * paired with.
     */
    fun buildBoth(recording: SignRecorder.Recording): Tensors {
        val (landmarks, present, n) = tensors(sample(recording))
        val xy = FeatureTransform.build(landmarks, present, n, MODEL_STEPS, useZ = false)
        val xyz = FeatureTransform.build(landmarks, present, n, MODEL_STEPS, useZ = true)
        return Tensors(xy.features, xyz.features, xy.mask)
    }

    /**
     * Mean absolute lag-1 velocity across the 24 steps, in normalised units
     * per model step.
     *
     * This is the motion the classifier actually receives — not motion in the
     * room. It is what the debug timeline shows, and what the recording-mode
     * comparison measures, because "the hands moved" and "the model was told
     * the hands moved" are different claims.
     */
    fun meanMotion(features: FloatArray): Float {
        val block = FeatureTransform.HANDS * FeatureTransform.LANDMARKS * 2 // 84
        var sum = 0.0
        var count = 0
        for (t in 1 until MODEL_STEPS) {
            val v = t * FeatureTransform.FEATURE_DIM + block
            for (i in 0 until block) {
                sum += kotlin.math.abs(features[v + i])
                count++
            }
        }
        return if (count == 0) 0f else (sum / count).toFloat()
    }

    /**
     * Frames whose features changed meaningfully from the previous step.
     *
     * Counts only steps where the normalised hand positions actually differ,
     * so a frozen hand held in view does not read as "motion" and a nearly
     * static sign is not penalised for being static.
     */
    fun movingSteps(features: FloatArray, epsilon: Float = 1e-4f): Int {
        val block = FeatureTransform.HANDS * FeatureTransform.LANDMARKS * 2
        var moving = 0
        for (t in 1 until MODEL_STEPS) {
            val a = t * FeatureTransform.FEATURE_DIM
            val b = (t - 1) * FeatureTransform.FEATURE_DIM
            var changed = false
            for (i in 0 until block) {
                if (kotlin.math.abs(features[a + i] - features[b + i]) > epsilon) {
                    changed = true
                    break
                }
            }
            if (changed) moving++
        }
        return moving
    }
}