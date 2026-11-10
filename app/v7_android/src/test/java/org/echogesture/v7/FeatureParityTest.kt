package org.echogesture.v7

import org.junit.Assert.assertArrayEquals
import org.junit.Test
import java.io.File
import javax.json.Json
import javax.json.JsonArray

/** JVM parity test: consumes the SAME JSON fixtures the Python reference
 *  (parity_reference.py) produces from src/features.py. Tolerance 1e-4 abs
 *  on features (float32 round-trip). Run: ./gradlew :app:testDebugUnitTest */
class FeatureParityTest {
    private fun load(name: String): JsonArray =
        Json.createReader(File("src/test/resources/$name.json").inputStream()).readArray()

    private fun toFloats(a: JsonArray): FloatArray {
        val out = FloatArray(a.size())
        for (i in a.indices) out[i] = a.getJsonNumber(i).doubleValue().toFloat()
        return out
    }

    @Test fun syntheticFeaturesMatchPython() = check("parity_synthetic")
    @Test fun chew2FeaturesMatchPython() = check("parity_chew2")

    private fun check(name: String) {
        val root = load(name)
        val lm: MutableList<Float> = mutableListOf()
        for (t in (root.getJsonObject(0).getJsonArray("landmarks"))) {
            for (h in t.asJsonArray()) for (l in h.asJsonArray()) for (c in l.asJsonArray()) lm.add(c.asJsonNumber().doubleValue().toFloat())
        }
        val present = toFloats(root.getJsonObject(0).getJsonArray("present"))
        val expected = toFloats(root.getJsonObject(0).getJsonArray("features"))
        val frames = root.getJsonObject(0).getJsonArray("landmarks").size()
        val got = FeatureTransform.buildFeatures(lm.toFloatArray(), present, frames, 24)
        assertArrayEquals(expected, got, 1e-4f)
    }
}
