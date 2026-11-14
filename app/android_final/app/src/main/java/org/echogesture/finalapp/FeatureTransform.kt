package org.echogesture.finalapp

import kotlin.math.sqrt

/**
 * Kotlin port of src/features.py — verified identical to the Python reference.
 *
 * Input: `landmarks` is a flattened (T, 2, 21, 3) float32 tensor laid out
 * [frame][handSlot][landmark][x,y,z]; `present` is a flattened (T, 2) float32
 * tensor; `srcFrames` is the raw frame count to resample from.
 *
 * Output: `FloatArray` of length `expectedFrames * FEATURE_DIM`, row-major with
 * the per-frame layout pos[84] | vel(lag-1)[84] | acc(lag-2)[84] | geom[3].
 *
 * Hand slots are canonical [Left=0, Right=1] (MediaPipe camera-POV handedness,
 * passed straight through — never swapped, matching src/landmark_frontend.py).
 * Per-frame normalisation uses a SHARED scale = mean wrist->middleMCP(9)
 * distance over present, valid hands (1.0 if none). Inter-hand geometry
 * [distance, dx, dy] is measured on the RAW landmarks before centring and is
 * zero unless both hands are present. Absent or degenerate hands are zeros.
 */
object FeatureTransform {
    const val HANDS = 2
    const val LANDMARKS = 21
    const val COORDS = 3          // x, y, z always present in the INPUT
    const val WRIST = 0
    const val MIDDLE_MCP = 9
    /** Width of the x,y feature vector: 2 hands x 21 x 2 coords x 3 blocks + 3. */
    const val FEATURE_DIM = HANDS * LANDMARKS * 2 * 3 + 3 // 255
    /** Width of the x,y,z feature vector: 2 hands x 21 x 3 coords x 3 blocks + 3. */
    const val FEATURE_DIM_XYZ = HANDS * LANDMARKS * 3 * 3 + 3 // 381

    /** Feature width for a member's `use_z`, matching src/features.py::feature_dim. */
    fun featureDim(useZ: Boolean): Int = if (useZ) FEATURE_DIM_XYZ else FEATURE_DIM

    fun buildFeatures(
        landmarks: FloatArray,
        present: FloatArray,
        srcFrames: Int,
        expectedFrames: Int = 24,
        useZ: Boolean = false
    ): FloatArray = build(landmarks, present, srcFrames, expectedFrames, useZ).features

    /** Feature matrix plus the attention mask, from one landmark pass. */
    data class Features(val features: FloatArray, val mask: FloatArray)

    /**
     * The attention mask the mask-input members were TRAINED with.
     *
     * `src/features.py` returns `(present.max(axis=1) > 0.5)` — over the
     * RESAMPLED presence, not the original frames. Doing it on the raw frames
     * would mark a different set of steps and silently feed the model a
     * different sequence than training did. Because the two feature widths are
     * built from the same landmarks, the mask is identical for both; it is
     * returned per build so it can never drift from the features beside it.
     */
    fun build(
        landmarks: FloatArray,
        present: FloatArray,
        srcFrames: Int,
        expectedFrames: Int = 24,
        useZ: Boolean = false
    ): Features {
        // Coordinate count is 2 (x,y) or 3 (x,y,z); the input always carries 3.
        val coords = if (useZ) 3 else 2
        val featureDim = featureDim(useZ)
        require(srcFrames >= 1) { "srcFrames must be >= 1, got $srcFrames" }
        require(landmarks.size == srcFrames * HANDS * LANDMARKS * COORDS) {
            "landmarks has ${landmarks.size} floats, expected ${srcFrames * HANDS * LANDMARKS * COORDS}"
        }
        require(present.size == srcFrames * HANDS) {
            "present has ${present.size} floats, expected ${srcFrames * HANDS}"
        }

        // 1) Linear resample landmarks + present along time to expectedFrames.
        //    Mirrors _resample() operating on xy (T,2,21,2) and present (T,2).
        val xy = FloatArray(expectedFrames * HANDS * LANDMARKS * coords)
        val pr = FloatArray(expectedFrames * HANDS)
        val srcPerFrame = HANDS * LANDMARKS * COORDS
        val xyPerFrame = HANDS * LANDMARKS * coords
        for (o in 0 until expectedFrames) {
            // Same index mapping as np.linspace(0, t-1, target).
            val idx = if (srcFrames <= 1) 0.0 else o.toDouble() * (srcFrames - 1) / (expectedFrames - 1)
            val lo = idx.toInt().coerceIn(0, srcFrames - 1)
            val hi = minOf(lo + 1, srcFrames - 1)
            val w = (idx - lo).toFloat()
            // Only x,y are carried into features; z is dropped (use_z=false).
            for (slot in 0 until HANDS) {
                for (l in 0 until LANDMARKS) {
                    for (c in 0 until coords) {
                        val li = lo * srcPerFrame + slot * LANDMARKS * COORDS + l * COORDS + c
                        val ri = hi * srcPerFrame + slot * LANDMARKS * COORDS + l * COORDS + c
                        val a = landmarks[li]
                        val b = landmarks[ri]
                        xy[o * xyPerFrame + slot * LANDMARKS * coords + l * coords + c] =
                            a * (1 - w) + b * w
                    }
                }
                pr[o * HANDS + slot] =
                    present[lo * HANDS + slot] * (1 - w) + present[hi * HANDS + slot] * w
            }
        }

        // 2) Per-frame: shared-scale root-centred hands + geometry from raw xy.
        // Geometry is ALWAYS x/y only (3 values), even in the z path: that is
        // what src/features.py does, and it is what makes 381 = 126*3 + 3.
        val block = HANDS * LANDMARKS * coords
        val norm = FloatArray(expectedFrames * block)
        val geom = FloatArray(expectedFrames * 3)
        for (t in 0 until expectedFrames) {
            val base = t * block
            val hands = Array(HANDS) { h ->
                xy.copyOfRange(base + h * LANDMARKS * coords, base + (h + 1) * LANDMARKS * coords)
            }
            val scales = arrayOfNulls<Float>(HANDS)
            for (h in 0 until HANDS) {
                if (pr[t * HANDS + h] > 0.5f) scales[h] = handScale(hands[h], coords)
            }
            val valid = scales.filterNotNull()
            val scale = if (valid.isNotEmpty()) valid.average().toFloat() else 1.0f
            for (h in 0 until HANDS) {
                if (pr[t * HANDS + h] > 0.5f && scales[h] != null) {
                    // Root-centre on this hand's wrist, then divide ALL coords
                    // by the shared scale. z is MediaPipe's relative depth; it
                    // is differenced and scaled like x/y, never re-based.
                    for (i in 0 until LANDMARKS * coords) {
                        val axis = i % coords
                        val sub = hands[h][WRIST * coords + axis]
                        norm[base + h * LANDMARKS * coords + i] = (hands[h][i] - sub) / scale
                    }
                }
            }
            if (pr[t * HANDS] > 0.5f && pr[t * HANDS + 1] > 0.5f) {
                val dx = (hands[1][WRIST * coords] - hands[0][WRIST * coords]) / scale
                val dy = (hands[1][WRIST * coords + 1] - hands[0][WRIST * coords + 1]) / scale
                geom[t * 3] = sqrt(dx * dx + dy * dy)
                geom[t * 3 + 1] = dx
                geom[t * 3 + 2] = dy
            }
        }

        // 3) Concatenate [norm, vel(lag-1), acc(lag-2), geom] -> (expectedFrames, 255).
        val out = FloatArray(expectedFrames * featureDim)
        for (t in 0 until expectedFrames) {
            val o = t * featureDim
            System.arraycopy(norm, t * block, out, o, block)
            for (i in 0 until block) {
                out[o + block + i] = if (t >= 1) norm[t * block + i] - norm[(t - 1) * block + i] else 0f
                out[o + 2 * block + i] = if (t >= 2) norm[t * block + i] - norm[(t - 2) * block + i] else 0f
            }
            System.arraycopy(geom, t * 3, out, o + 3 * block, 3)
        }
        // Mask over the RESAMPLED presence, exactly as src/features.py does:
        // mask[t] = 1 when either hand's interpolated presence exceeds 0.5.
        val mask = FloatArray(expectedFrames)
        for (t in 0 until expectedFrames) {
            val a = pr[t * HANDS]
            val b = pr[t * HANDS + 1]
            mask[t] = if (a > 0.5f || b > 0.5f) 1f else 0f
        }

        return Features(out, mask)
    }

    /**
     * Wrist-to-middle-MCP distance, over AS MANY COORDINATES AS THE FEATURES USE.
     *
     * src/features.py calls `np.linalg.norm(hand[MIDDLE_MCP] - hand[WRIST])` on
     * whatever it was handed. In the z path that is a 3-vector, so the norm
     * INCLUDES z. The five z-members were trained with that scale, so using
     * x,y only here would rescale every one of their features by a different
     * factor per frame — the model still runs and still returns a plausible
     * label, which is exactly why this needed a numerical parity test to find.
     * Measured before the fix: feature max abs diff 1.047.
     *
     * The docstring in features.py still says "one 21x2 hand"; it is stale.
     * The code is the contract.
     */
    private fun handScale(hand: FloatArray, coords: Int): Float? {
        var sum = 0f
        for (axis in 0 until coords) {
            val d = hand[MIDDLE_MCP * coords + axis] - hand[WRIST * coords + axis]
            sum += d * d
        }
        val d = sqrt(sum)
        return if (d.isFinite() && d > 1e-6f) d else null
    }
}