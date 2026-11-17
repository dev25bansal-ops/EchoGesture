package org.echogesture.finalapp

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Proves the profile record actually reaches the diagnostics file.
 *
 * This exists because the profiler was once ported into the app without ever
 * being called: it compiled, it was in the APK, and it wrote nothing. An absent
 * `profile` line then looks identical to "the analyser measured nothing", which
 * is the wrong conclusion to draw from silence.
 */
@RunWith(AndroidJUnit4::class)
class DiagnosticsProfileWriteTest {

    @Test
    fun profileRecordIsWrittenToTheFile() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val log = DiagnosticsLog(ctx)
        log.clear()

        val profiler = AnalysisProfiler()
        // 30 fps for a second, as a real camera would deliver.
        repeat(30) { i -> profiler.record(i * 33L, convertDurationMs = 4, detectDurationMs = 18, sensorTimestampNs = 0) }
        log.recordProfiler(profiler.snapshot())

        val text = log.read()
        assertTrue("file should contain a profile line, got:\n$text", text.contains("profile fps="))
        // The exact rate is asserted loosely on purpose: the first sample has no
        // preceding interval, so a 30 Hz stream reports one fewer gap and lands
        // just under 30. What matters is that a real number is present, not 0.
        val fps = Regex("profile fps=([0-9.]+)").find(text)?.groupValues?.get(1)?.toDoubleOrNull()
        assertTrue("profile fps should be a positive number, got:\n$text", fps != null && fps > 5.0)
        assertTrue("profile should name the limiting side", text.contains("limit="))
        log.clear()
    }

    /**
     * A capture record and a profile record must coexist.
     *
     * Both append to one file and each rewrites it. If the second write
     * dropped the first, a real session would lose the capture line that says
     * what the model was given.
     */
    @Test
    fun captureAndProfileRecordsCoexist() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val log = DiagnosticsLog(ctx)
        log.clear()

        val capture = HandTracker.Capture(emptyList(), 1200L, 25f, 0)
        log.record(capture, null, 0L)

        val profiler = AnalysisProfiler()
        repeat(20) { i -> profiler.record(i * 40L, 5, 20, 0) }
        log.recordProfiler(profiler.snapshot())

        val text = log.read()
        assertTrue("capture line must survive the profile write:\n$text", text.contains("frames=0"))
        assertTrue("profile line must be present:\n$text", text.contains("profile fps="))
        log.clear()
    }

    /** A note records an absence as a fact rather than leaving a silent gap. */
    @Test
    fun notesAreWrittenAndReadable() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val log = DiagnosticsLog(ctx)
        log.clear()
        log.recordNote("profile_unavailable tracker=null")
        assertTrue(log.read().contains("note profile_unavailable"))
        log.clear()
    }
}