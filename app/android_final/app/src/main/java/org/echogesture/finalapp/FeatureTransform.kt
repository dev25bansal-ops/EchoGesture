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
    const val COORDS = 3          // x, y, z in the input landmark slot
    const val WRIST = 0
    const val MIDDLE_MCP = 9
    const val FEATURE_DIM = HANDS * LANDMARKS * 2 * 3 + 3 // 255

    fun buildFeatures(
        landmarks: FloatArray,
        present: FloatArray,
        srcFrames: Int,
        expectedFrames: Int = 24
    ): FloatArray {
        require(srcFrames >= 1) { "srcFrames must be >= 1, got $srcFrames" }
        require(landmarks.size == srcFrames * HANDS * LANDMARKS * COORDS) {
            "landmarks has ${landmarks.size} floats, expected ${srcFrames * HANDS * LANDMARKS * COORDS}"
        }
        require(present.size == srcFrames * HANDS) {
            "present has ${present.size} floats, expected ${srcFrames * HANDS}"
        }

        // 1) Linear resample landmarks + present along time to expectedFrames.
        //    Mirrors _resample() operating on xy (T,2,21,2) and present (T,2).
        val xy = FloatArray(expectedFrames * HANDS * LANDMARKS * 2)
        val pr = FloatArray(expectedFrames * HANDS)
        val srcPerFrame = HANDS * LANDMARKS * COORDS
        val xyPerFrame = HANDS * LANDMARKS * 2
        for (o in 0 until expectedFrames) {
            // Same index mapping as np.linspace(0, t-1, target).
            val idx = if (srcFrames <= 1) 0.0 else o.toDouble() * (srcFrames - 1) / (expectedFrames - 1)
            val lo = idx.toInt().coerceIn(0, srcFrames - 1)
            val hi = minOf(lo + 1, srcFrames - 1)
            val w = (idx - lo).toFloat()
            // Only x,y are carried into features; z is dropped (use_z=false).
            for (slot in 0 until HANDS) {
                for (l in 0 until LANDMARKS) {
                    for (c in 0 until 2) {
                        val li = lo * srcPerFrame + slot * LANDMARKS * COORDS + l * COORDS + c
                        val ri = hi * srcPerFrame + slot * LANDMARKS * COORDS + l * COORDS + c
                        val a = landmarks[li]
                        val b = landmarks[ri]
                        xy[o * xyPerFrame + slot * LANDMARKS * 2 + l * 2 + c] =
                            a * (1 - w) + b * w
                    }
                }
                pr[o * HANDS + slot] =
                    present[lo * HANDS + slot] * (1 - w) + present[hi * HANDS + slot] * w
            }
        }

        // 2) Per-frame: shared-scale root-centred hands + geometry from raw xy.
        val block = HANDS * LANDMARKS * 2 // 84
        val norm = FloatArray(expectedFrames * block)
        val geom = FloatArray(expectedFrames * 3)
        for (t in 0 until expectedFrames) {
            val base = t * block
            val hands = Array(HANDS) { h -> xy.copyOfRange(base + h * LANDMARKS * 2, base + (h + 1) * LANDMARKS * 2) }
            val scales = arrayOfNulls<Float>(HANDS)
            for (h in 0 until HANDS) {
                if (pr[t * HANDS + h] > 0.5f) scales[h] = handScale(hands[h])
            }
            val valid = scales.filterNotNull()
            val scale = if (valid.isNotEmpty()) valid.average().toFloat() else 1.0f
            for (h in 0 until HANDS) {
                if (pr[t * HANDS + h] > 0.5f && scales[h] != null) {
                    val wx = hands[h][WRIST * 2]
                    val wy = hands[h][WRIST * 2 + 1]
                    for (i in 0 until LANDMARKS * 2) {
                        val sub = if (i % 2 == 0) wx else wy
                        norm[base + h * LANDMARKS * 2 + i] = (hands[h][i] - sub) / scale
                    }
                }
            }
            if (pr[t * HANDS] > 0.5f && pr[t * HANDS + 1] > 0.5f) {
                val dx = (hands[1][WRIST * 2] - hands[0][WRIST * 2]) / scale
                val dy = (hands[1][WRIST * 2 + 1] - hands[0][WRIST * 2 + 1]) / scale
                geom[t * 3] = sqrt(dx * dx + dy * dy)
                geom[t * 3 + 1] = dx
                geom[t * 3 + 2] = dy
            }
        }

        // 3) Concatenate [norm, vel(lag-1), acc(lag-2), geom] -> (expectedFrames, 255).
        val out = FloatArray(expectedFrames * FEATURE_DIM)
        for (t in 0 until expectedFrames) {
            val o = t * FEATURE_DIM
            System.arraycopy(norm, t * block, out, o, block)
            for (i in 0 until block) {
                out[o + block + i] = if (t >= 1) norm[t * block + i] - norm[(t - 1) * block + i] else 0f
                out[o + 2 * block + i] = if (t >= 2) norm[t * block + i] - norm[(t - 2) * block + i] else 0f
            }
            System.arraycopy(geom, t * 3, out, o + 3 * block, 3)
        }
        return out
    }

    /** Wrist-to-middle-MCP distance for one 21x2 hand, or null if degenerate. */
    private fun handScale(xy21x2: FloatArray): Float? {
        val dx = xy21x2[MIDDLE_MCP * 2] - xy21x2[WRIST * 2]
        val dy = xy21x2[MIDDLE_MCP * 2 + 1] - xy21x2[WRIST * 2 + 1]
        val d = sqrt(dx * dx + dy * dy)
        return if (d.isFinite() && d > 1e-6f) d else null
    }
}