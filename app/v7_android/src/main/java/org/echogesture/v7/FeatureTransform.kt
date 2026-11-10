package org.echogesture.v7

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Kotlin port of src/features.py == 1:1 =========================
 * Expected input (same as Python): landmarks (T, 2, 21, 3) float32,
 *   present (T, 2) float32, expectedFrames = 24.
 * Output: FloatArray flattened (24*255) row-major, layout
 *   pos[84] | vel[84] | acc[84] | geom[3] per frame.
 * Ordering per hand slot: [Left=0, Right=1]; within a hand slot the
 * 84 values are landmark-major: landmark i contributes (x, y) pairs
 * -> index = hand*42 + landmark*2 + coord. This matches numpy
 * reshape(T,-1) on (2,21,2), which is C-order over (hand, landmark, coord).
 * Per-frame normalisation uses a SHARED scale = mean of the
 * wrist->middleMCP(9) distances of present, valid hands (1.0 if none).
 * Absent/degenerate hands are all-zero.
 */
object FeatureTransform {
    const val HANDS = 2
    const val LANDMARKS = 21
    const val WRIST = 0
    const val MIDDLE_MCP = 9
    const val FEATURE_DIM = HANDS * LANDMARKS * 2 * 3 + 3 // 255

    private fun handScale(xy: Array<FloatArray>, hand: Int): Float? {
        // xy[hand] is (21*2) flattened
        val w = hand * LANDMARKS * 2
        val dx = xy[hand][(MIDDLE_MCP) * 2] - xy[hand][WRIST * 2]
        val dy = xy[hand][(MIDDLE_MCP) * 2 + 1] - xy[hand][WRIST * 2 + 1]
        val d = sqrt(dx * dx + dy * dy)
        return if (d.isFinite() && d > 1e-6f) d else null
    }

    /** landmarks: ((T x 2 x 21 x 3) flattened), present: (T x 2). Returns (24*255). */
    fun buildFeatures(landmarks: FloatArray, present: FloatArray, srcFrames: Int,
                      expectedFrames: Int = 24): FloatArray {
        // 1) resample landmarks+present linearly to expectedFrames
        //    (mirrors _resample on xy (T,2,21,2) and present (T,2))
        val xy = FloatArray(expectedFrames * HANDS * LANDMARKS * 2)
        val pr = FloatArray(expectedFrames * HANDS)
        for (o in 0 until expectedFrames) {
            val idx = if (srcFrames <= 1) 0.0 else o.toDouble() * (srcFrames - 1) / (expectedFrames - 1)
            val lo = idx.toInt().coerceIn(0, srcFrames - 1)
            val hi = minOf(lo + 1, srcFrames - 1)
            val w = (idx - lo).toFloat()
            for (i in 0 until HANDS * LANDMARKS * 2) {
                val a = landmarks[(lo * HANDS * LANDMARKS * 3) + (i / 2) * 3 + (i % 2)]
                val b = landmarks[(hi * HANDS * LANDMARKS * 3) + (i / 2) * 3 + (i % 2)]
                xy[o * HANDS * LANDMARKS * 2 + i] = a * (1 - w) + b * w
            }
            for (h in 0 until HANDS) {
                val a = present[lo * HANDS + h]; val b = present[hi * HANDS + h]
                pr[o * HANDS + h] = a * (1 - w) + b * w
            }
        }

        // 2) per-frame: shared-scale root-centred normalised hands + geometry from RAW xy
        val norm = FloatArray(expectedFrames * HANDS * LANDMARKS * 2)
        val geom = FloatArray(expectedFrames * 3)
        for (t in 0 until expectedFrames) {
            val base = t * HANDS * LANDMARKS * 2
            val hands = Array(HANDS) { h -> xy.copyOfRange(base + h * LANDMARKS * 2, base + (h + 1) * LANDMARKS * 2) }
            val scales = arrayOfNulls<Float>(HANDS)
            for (h in 0 until HANDS) if (pr[t * HANDS + h] > 0.5f) scales[h] = handScale(hands, h)
            val valid = scales.filterNotNull()
            val scale = if (valid.isNotEmpty()) valid.average().toFloat() else 1.0f
            for (h in 0 until HANDS) {
                if (pr[t * HANDS + h] > 0.5f && scales[h] != null) {
                    val wx = hands[h][WRIST * 2]; val wy = hands[h][WRIST * 2 + 1]
                    for (i in 0 until LANDMARKS * 2)
                        norm[base + h * LANDMARKS * 2 + i] = (hands[h][i] - if (i % 2 == 0) wx else wy) / scale
                }
            }
            if (pr[t * HANDS] > 0.5f && pr[t * HANDS + 1] > 0.5f) {
                val dx = (hands[1][WRIST * 2] - hands[0][WRIST * 2]) / scale
                val dy = (hands[1][WRIST * 2 + 1] - hands[0][WRIST * 2 + 1]) / scale
                geom[t * 3] = sqrt(dx * dx + dy * dy); geom[t * 3 + 1] = dx; geom[t * 3 + 2] = dy
            }
        }

        // 3) concatenate [norm, vel(lag1), acc(lag2), geom] -> (T, 255)
        val out = FloatArray(expectedFrames * FEATURE_DIM)
        val block = HANDS * LANDMARKS * 2 // 84
        for (t in 0 until expectedFrames) {
            val o = t * FEATURE_DIM
            System.arraycopy(norm, t * block, out, o, block)
            for (i in 0 until block) {
                out[o + block + i] = (if (t >= 1) norm[t * block + i] - norm[(t - 1) * block + i] else 0f)
                out[o + 2 * block + i] = (if (t >= 2) norm[t * block + i] - norm[(t - 2) * block + i] else 0f)
            }
            System.arraycopy(geom, t * 3, out, o + 3 * block, 3)
        }
        return out
    }
}
