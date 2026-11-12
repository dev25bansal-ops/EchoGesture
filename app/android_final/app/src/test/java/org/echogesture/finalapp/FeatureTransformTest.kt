package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertArrayEquals
import org.junit.Test

/**
 * JVM feature-parity tests. Fixture tests are pure Kotlin (no model or Android
 * binaries): they consume the same numbers the host reference produced from
 * src/features.py and require buildFeatures to match within float32 tolerance.
 */
class FeatureTransformTest {

    private fun loadFixture(name: String): Triple<FloatArray, FloatArray, FloatArray> {
        val lines = javaClass.classLoader!!.getResource("$name.txt")
            ?.readText()
            ?.lineSequence()
            ?.toList()
            ?: error("missing fixture $name.txt")
        val t = lines[0].trim().split(' ')[0].toInt()
        val lm = parseFloats(lines[1], t * 2 * 21 * 3)
        val present = parseFloats(lines[2], t * 2)
        val expectedFeatures = parseFloats(lines[3], 24 * 255)
        return Triple(lm, present, expectedFeatures)
    }

    private fun parseFloats(line: String, expectedCount: Int): FloatArray {
        val values = line.trim().split(' ').filter { it.isNotEmpty() }
        assertEquals("float count", expectedCount, values.size)
        return FloatArray(values.size) { values[it].toFloat() }
    }

    private fun maxAbsDiff(a: FloatArray, b: FloatArray): Float {
        var m = 0f
        for (i in a.indices) {
            val d = kotlin.math.abs(a[i] - b[i])
            if (d > m) m = d
        }
        return m
    }

    @Test
    fun syntheticFeaturesMatchPython() {
        val (lm, present, expected) = loadFixture("parity_synthetic")
        val t = present.size / 2
        val got = FeatureTransform.buildFeatures(lm, present, t, 24)
        assertArrayEquals("synthetic features drift from src/features.py", expected, got, 1e-4f)
    }

    @Test
    fun chew2FeaturesMatchPython() {
        val (lm, present, expected) = loadFixture("parity_chew2")
        val t = present.size / 2
        val got = FeatureTransform.buildFeatures(lm, present, t, 24)
        assertArrayEquals("chew2 features drift from src/features.py", expected, got, 1e-4f)
    }

    // ---- Edge cases, mirroring semantics documented in src/features.py ----

    @Test
    fun allAbsentYieldsZeros() {
        // 1 frame, both hands absent -> all features zero (scale fallback 1.0,
        // no hand contributes, geometry zero).
        val lm = FloatArray(1 * 2 * 21 * 3)
        val present = FloatArray(2) { 0f }
        val got = FeatureTransform.buildFeatures(lm, present, 1, 24)
        assertArrayEquals("absent hands must produce zeros", FloatArray(24 * 255), got, 0f)
    }

    @Test
    fun velocityAndAccelerationAreLagCorrect() {
        // Two identical present frames produce zero velocity and acceleration in
        // the resampled output (positions equal across the resample window).
        val t = 2
        val lm = FloatArray(t * 2 * 21 * 3)
        // Right-hand wrist at x=0.1 -> middleMCP x=0.2 gives a nonzero scale;
        // identical across both frames so norm is constant.
        for (f in 0 until t) {
            val base = f * 2 * 21 * 3
            lm[base + 1 * 21 * 3 + 0 * 3 + 0] = 0.1f
            lm[base + 1 * 21 * 3 + 9 * 3 + 0] = 0.2f
        }
        val present = FloatArray(t * 2) { f -> if (f % 2 == 1) 1f else 0f }
        val got = FeatureTransform.buildFeatures(lm, present, t, 24)
        val block = 84
        for (s in 0 until 24) {
            val o = s * 255
            for (i in 0 until block) {
                val vel = got[o + block + i]
                val acc = got[o + 2 * block + i]
                if (s >= 1) assertEquals("velocity lag-1 must be zero for constant pos", 0f, vel, 1e-5f)
                if (s >= 2) assertEquals("acceleration lag-2 must be zero for constant pos", 0f, acc, 1e-5f)
            }
        }
    }

    @Test
    fun featureDimensionIs255() {
        assertEquals(255, FeatureTransform.FEATURE_DIM)
    }
}