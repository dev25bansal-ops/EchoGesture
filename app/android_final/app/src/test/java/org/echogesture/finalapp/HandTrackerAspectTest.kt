package org.echogesture.finalapp

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * MediaPipe normalises landmark x by the frame's own width and y by its own
 * height, so the same physical hand yields different normalised coordinates on
 * differently-shaped frames. The checkpoints were trained on 640x480 clips; the
 * app is locked to portrait and hands the detector 480x640. Unless the camera
 * path is mapped back into corpus normalisation, every hand reaches the model
 * stretched.
 *
 * This is not cosmetic. With the nine shipped members, inflicting this warp on
 * the 1,251-clip test set costs 85.08% -> 82.66% top-1; on a real 32-frame
 * camera capture, undoing it moved member agreement from 1/9 to 5/9 and the
 * top-1 margin from 0.061 to 0.202.
 *
 * A uniform overall scale is deliberately NOT corrected: `FeatureTransform`
 * root-centres on the wrist and divides by wrist->middle-MCP, so only the
 * RATIO between the two axis factors can reach the model.
 */
class HandTrackerAspectTest {

    private fun hand(): FloatArray = FloatArray(63) { i ->
        when (i % 3) {
            0 -> 0.30f + (i / 3) * 0.01f
            1 -> 0.55f + (i / 3) * 0.01f
            else -> (i / 3) * 0.002f
        }
    }

    @Test
    fun aCorpusShapedFrameIsLeftExactlyAlone() {
        // 640x480 is the geometry the checkpoints were trained on, so the map
        // into corpus normalisation is the identity and must perturb nothing.
        val hand = hand()
        val before = hand.copyOf()
        HandTracker.mapToCorpusAspect(hand, 640, 480)
        assertArrayEquals("a 640x480 frame must pass through unchanged", before, hand, 0.0f)
    }

    @Test
    fun aPortraitFrameIsMappedBackToCorpusAspect() {
        // A hand reported on a 480x640 portrait frame carries an anisotropy of
        // (480/640)/(640/480) = 0.5625 relative to the corpus. After the map its
        // x:y ratio must match what the same hand would have had in a clip.
        val landscape = hand()
        val asPortrait = FloatArray(63) { i ->
            when (i % 3) {
                0 -> landscape[i] * (640f / 480f)   // x looked wider
                1 -> landscape[i] * (480f / 640f)   // y looked shorter
                else -> landscape[i]
            }
        }
        HandTracker.mapToCorpusAspect(asPortrait, 480, 640)

        // The map is exact up to one uniform factor (W/640, H/480) = (0.75,
        // 1.3333), so compare shape rather than raw values: every component of
        // the corrected hand must be the corresponding one scaled by the SAME
        // constant. Anything else would mean the two axes were treated
        // differently, which is the defect this guards.
        val ratios = (0 until 63)
            .filter { abs(landscape[it]) > 1e-4f }
            .map { asPortrait[it] / landscape[it] }
        val first = ratios.first()
        for (r in ratios) {
            assertTrue("x:y correction was not uniform (saw $r and $first)", abs(r - first) < 1e-5f)
        }
    }

    @Test
    fun theTwoAxesAreCorrectedByTheFrameAspectRatio() {
        // The whole defect is the RATIO between the axes. On portrait it must be
        // (480/640)/(640/480) = 0.5625 relative to a 640x480 frame.
        val zero = floatArrayOf(0f, 0f, 0f)
        val one = floatArrayOf(1f, 1f, 0f)
        HandTracker.mapToCorpusAspect(zero, 480, 640)
        HandTracker.mapToCorpusAspect(one, 480, 640)
        val xRatio = one[0] - zero[0]
        val yRatio = one[1] - zero[1]
        assertEquals("portrait x:y correction must be 0.75/1.3333",
            0.5625f, xRatio / yRatio, 1e-5f)
    }

    @Test
    fun aSquareFrameIsCorrectedToo() {
        // 960x960 is the shape the Vivo V2404 actually delivered when asked for
        // 640x480. It is not square in corpus terms, so it must not pass through.
        val zero = floatArrayOf(0f, 0f, 0f)
        val one = floatArrayOf(1f, 1f, 0f)
        HandTracker.mapToCorpusAspect(zero, 960, 960)
        HandTracker.mapToCorpusAspect(one, 960, 960)
        val xRatio = one[0] - zero[0]
        val yRatio = one[1] - zero[1]
        assertEquals("960x960 must be corrected to 0.75, not left alone",
            0.75f, xRatio / yRatio, 1e-5f)
    }

    @Test
    fun zIsNeverTouched() {
        // z is MediaPipe's relative depth, in units of the hand rather than of
        // the frame. Scaling it with the frame aspect would be a different and
        // wrong operation, so it must survive untouched on every path.
        val hand = FloatArray(63) { i -> (i % 3).toFloat() * 0.25f }
        val zBefore = (2 until 63 step 3).map { hand[it] }
        HandTracker.mapToCorpusAspect(hand, 480, 640)
        val zAfter = (2 until 63 step 3).map { hand[it] }
        assertEquals(zBefore, zAfter)
    }

    @Test
    fun degenerateDimensionsAreIgnoredRatherThanDividingByZero() {
        // A frame that has not been sized yet reports 0; it must not poison the
        // landmarks with NaN/Inf -- features.py refuses non-finite input and the
        // capture would be lost for a reason that has nothing to do with the sign.
        val hand = hand()
        val before = hand.copyOf()
        val out = HandTracker.mapToCorpusAspect(hand, 0, 0)
        assertArrayEquals(before, out, 0.0f)
        assertTrue(out.all { it.isFinite() })
    }
}