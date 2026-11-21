package org.echogesture.finalapp

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The diagnostics export must produce a shareable file, and must never invent one.
 *
 * Export exists so a consented test session can be handed over without a cable.
 * Two things therefore matter: an empty record produces NO file (rather than a
 * blank one that looks like "nothing was wrong"), and a real record produces a
 * file whose header states what it does and does not contain.
 */
@RunWith(AndroidJUnit4::class)
class DiagnosticsExportTest {

    private fun log(): DiagnosticsLog {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        return DiagnosticsLog(ctx)
    }

    /** With nothing recorded, export reports nothing rather than sharing a blank. */
    @Test
    fun emptyRecordExportsNothing() {
        val d = log()
        d.clear()
        assertNull("an empty record must not produce a file", d.exportCopy("test"))
    }

    @Test
    fun disablingAndClearingRemovesBothFilesAndBlocksLateWrites() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        var enabled = true
        val log = DiagnosticsLog(context) { enabled }
        val capture = HandTracker.Capture(emptyList(), 1L, 1f, 0)
        log.clear()
        log.record(capture, null, 0L, fromClip = true)
        val exported = checkNotNull(log.exportCopy("test"))
        assertTrue(exported.readText().contains("source=clip"))
        enabled = false
        log.clear()
        log.record(capture, null, 0L)
        assertNull(log.exportCopy("test"))
        assertTrue(!exported.exists())
        assertTrue(!java.io.File(log.path()).exists())
    }

    /** A real record exports, and the copy carries the privacy header. */
    @Test
    fun recordExportsWithPrivacyHeader() {
        val d = log()
        d.clear()
        val capture = HandTracker.Capture(emptyList(), 2100L, 15.2f, 18)
        d.record(capture, null, 0L, DiagnosticsLog.Attempt(
            endReason = "QUIET",
            startedAtMs = 1000L,
            endedAtMs = 2800L,
            maskSteps = 20,
            bothHandsFrames = 14,
            longestDropoutFrames = 2,
            slotFlips = 0
        ))

        val out = d.exportCopy("2026-11-19 11:10")
        assertNotNull("a populated record must export", out)
        val text = out!!.readText()

        assertTrue("must say what it is", text.contains("Exported from EchoGesture settings"))
        assertTrue("must state no images", text.contains("No images"))
        assertTrue("must carry the attempt fields", text.contains("end_reason=QUIET"))
        assertTrue("must carry segment timing", text.contains("seg_ms=1800"))
        assertTrue("must carry mask coverage", text.contains("mask_steps=20/24"))
        assertTrue("must carry dropout", text.contains("longest_dropout=2"))

        // Nothing that could identify a person or reconstruct their hands.
        // The header legitimately says "no landmarks", so look for the data
        // itself rather than the word: a landmark array would appear as a run of
        // decimal numbers, which no statistics line produces.
        val longNumberRun = Regex("(-?\\d+\\.\\d+[,\\s]+){8,}").containsMatchIn(text)
        assertTrue("record must carry no landmark arrays", !longNumberRun)
        runCatching { out.delete() }
        d.clear()
    }

    /**
     * The exported copy lives under files/export/, which is the one directory the
     * FileProvider exposes. A copy written anywhere else could not be shared, and
     * widening the provider to cover other directories would expose more than
     * this feature needs.
     */
    @Test
    fun exportCopyLandsInTheSharedDirectory() {
        val d = log()
        d.clear()
        d.record(HandTracker.Capture(emptyList(), 1L, 1f, 0), null, 0L)
        val out = d.exportCopy("test")
        assertNotNull(out)
        assertTrue(
            "export must sit under files/export, was ${out!!.absolutePath}",
            out.parentFile?.name == "export"
        )
        runCatching { out.delete() }
        d.clear()
    }
}
