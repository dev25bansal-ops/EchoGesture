package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for the live analysis-rate measurement.
 *
 * This exists because the project previously reported "5-10 fps" from a figure
 * that was actually the model's sampled-window size, capped at 32. The whole
 * point of this class is to make the real rate measurable, so the tests pin the
 * arithmetic that turns frame timings into a rate and a drop count.
 */
class AnalysisProfilerTest {

    private fun record(
        p: AnalysisProfiler,
        arrivalMs: Long,
        convert: Long = 5,
        detect: Long = 20,
        sensorNs: Long = 0
    ) = p.record(arrivalMs, convert, detect, sensorNs)

    @Test
    fun `no frames reports no data`() {
        val snap = AnalysisProfiler().snapshot()
        assertEquals(0, snap.frames)
        assertEquals(AnalysisProfiler.Limit.NO_DATA, snap.limit)
        assertEquals(0.0, snap.effectiveFps, 0.001)
    }

    /** Frames arriving every 33 ms must read as ~30 fps. */
    @Test
    fun `steady 33 ms arrivals read as 30 fps`() {
        val p = AnalysisProfiler()
        repeat(40) { i -> record(p, arrivalMs = i * 33L) }
        val snap = p.snapshot()
        assertEquals(40, snap.frames)
        assertEquals(30.0, snap.effectiveFps, 1.0)
    }

    /** Frames arriving every 100 ms are 10 fps, not "32 samples". */
    @Test
    fun `steady 100 ms arrivals read as 10 fps`() {
        val p = AnalysisProfiler()
        repeat(30) { i -> record(p, arrivalMs = i * 100L) }
        assertEquals(10.0, p.snapshot().effectiveFps, 0.5)
    }

    /**
     * A camera that produces frames faster than the analyser consumes them
     * must show up as dropped frames, derived from the sensor timestamps.
     *
     * The first timestamp must be non-zero: `0` is the code's sentinel for
     * "this device did not report a sensor time", so a zero start would mean
     * there is no baseline to measure a gap from. Real cameras report a real
     * timestamp, which is what this reproduces.
     */
    @Test
    fun `sensor gaps count dropped frames`() {
        val p = AnalysisProfiler()
        // 33 ms nominal. Sensor times 1_000 ms then 1_330 ms: 10 intervals
        // elapsed, so 9 frames were produced and never analysed.
        record(p, 0, sensorNs = 1_000_000_000L)
        record(p, 330, sensorNs = 1_330_000_000L)
        val snap = p.snapshot()
        assertEquals(10, snap.delivered)
        assertEquals(9, snap.dropped)
    }

    /** A first call with no sensor timestamp must not fabricate a baseline. */
    @Test
    fun `zero first sensor timestamp establishes no baseline`() {
        val p = AnalysisProfiler()
        record(p, 0, sensorNs = 0)
        record(p, 330, sensorNs = 330_000_000L)
        // Only the second call set lastSensorMs, so no gap can be measured yet.
        assertEquals(0L, p.snapshot().delivered)
        assertEquals(0L, p.snapshot().dropped)
    }

    /** A device that reports no sensor timestamp must not invent drops. */
    @Test
    fun `missing sensor timestamps do not fabricate drops`() {
        val p = AnalysisProfiler()
        repeat(20) { i -> record(p, i * 50L, sensorNs = 0) }
        assertEquals(0L, p.snapshot().dropped)
    }

    /** When analysis fills the arrival interval, the analyser is the limit. */
    @Test
    fun `slow analysis is attributed to the analyser`() {
        val p = AnalysisProfiler()
        repeat(20) { i -> record(p, i * 33L, convert = 5, detect = 30) }
        assertEquals(AnalysisProfiler.Limit.ANALYSER, p.snapshot().limit)
    }

    /** Fast analysis with no drops means the camera is the limit. */
    @Test
    fun `fast analysis with no drops is attributed to the camera`() {
        val p = AnalysisProfiler()
        repeat(20) { i -> record(p, i * 66L, convert = 3, detect = 8) }
        assertEquals(AnalysisProfiler.Limit.CAMERA, p.snapshot().limit)
    }

    /** Stage timings must survive into the snapshot for the split to be useful. */
    @Test
    fun `stage timings are reported separately`() {
        val p = AnalysisProfiler()
        repeat(20) { i -> record(p, i * 50L, convert = 7, detect = 22) }
        val snap = p.snapshot()
        assertEquals(7L, snap.convertP50)
        assertEquals(22L, snap.detectP50)
        assertEquals(29L, snap.analysisP50)
    }

    /** The buffer is bounded: a long session must not grow without limit. */
    @Test
    fun `ring buffer is bounded`() {
        val p = AnalysisProfiler(capacity = 16)
        repeat(500) { i -> record(p, i * 33L) }
        assertTrue("frames stays within capacity", p.snapshot().frames <= 16)
    }

    /** reset() must clear the window so a new session starts clean. */
    @Test
    fun `reset clears the window`() {
        val p = AnalysisProfiler()
        repeat(20) { i -> record(p, i * 33L) }
        p.reset()
        assertEquals(0, p.snapshot().frames)
        assertEquals(AnalysisProfiler.Limit.NO_DATA, p.snapshot().limit)
    }
}