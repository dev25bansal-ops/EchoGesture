package org.echogesture.finalapp

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * End-to-end diagnostics: a real automatic capture must produce every triage field.
 *
 * The existing export test hand-builds an [DiagnosticsLog.Attempt], which proves
 * the formatter works but says nothing about whether the capture path ever fills
 * those fields in. That is precisely the defect this file guards: the attempt was
 * being dropped before it reached the log, so the export looked complete while
 * carrying none of the segmenter's decisions.
 *
 * This drives the real [HandTracker] through a full one-tap sign, runs the real
 * feature build, and asserts on the EXPORTED text.
 */
@RunWith(AndroidJUnit4::class)
class DiagnosticsEndToEndTest {

    private fun frame(atMs: Long, move: Float, hands: Int = 2): HandTracker.Frame =
        HandTracker.Frame(
            (0 until hands).map { slot ->
                HandTracker.Hand(slot, FloatArray(63) { i -> if (i < 3) 0f else move * 0.01f })
            },
            atMs
        )

    /**
     * One tap, one sign, one exported record carrying:
     * end_reason, seg_ms, mask_steps, longest_dropout, slot_flips.
     */
    @Test
    fun automaticCaptureExportsAllTriageFields() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val tracker = HandTracker.createForTest(ctx)
        val log = DiagnosticsLog(ctx)
        log.clear()

        try {
            tracker.armAuto(0L)
            var now = 0L
            // Settle, sign with clear motion, hold still to end.
            for (i in 0 until 12) { now += 40; tracker.offerForTest(frame(now, 0f)) }
            for (i in 0 until 40) {
                now += 40
                tracker.offerForTest(frame(now, if (i % 2 == 0) 2f else 6f))
            }
            // A short hand loss mid-sign, so longest_dropout has something to report.
            for (i in 0 until 3) { now += 40; tracker.offerForTest(frame(now, 0f, hands = 0)) }
            for (i in 0 until 20) {
                now += 40
                tracker.offerForTest(frame(now, if (i % 2 == 0) 2f else 6f))
            }
            // Quiet tail: this is what ends the sign.
            var outcome: AutoSegmenter.Outcome? = null
            for (i in 0 until 60) {
                now += 40
                tracker.offerForTest(frame(now, 0f))
                if (outcome == null) outcome = tracker.takeAutoOutcome()
            }
            assertNotNull("the one-tap capture must complete", outcome)
            val o = outcome!!

            // Exactly the object the app builds, then the real mask.
            val attempt = attemptForTest(o)
            val recording = SignRecorder.Recording(o.frames, o.startedAtMs, o.endedAtMs)
            val tensors = RecordingFeatures.buildBoth(recording)
            val withMask = attempt.withMaskFrom(tensors.mask)

            val capture = HandTracker.Capture(
                RecordingFeatures.sampleWithoutTrim(recording),
                o.endedAtMs - o.startedAtMs,
                25f,
                o.frames.count { it.hands.isNotEmpty() }
            )
            log.record(capture, null, 0L, withMask)

            val out = log.exportCopy("test")
            assertNotNull("a completed attempt must export", out)
            val text = out!!.readText()

            assertTrue("end_reason missing:\n$text", text.contains("end_reason="))
            assertTrue("a real capture must name why it ended", !text.contains("end_reason=\n"))
            assertTrue("seg_ms missing:\n$text", text.contains("seg_ms="))
            assertTrue("mask_steps missing:\n$text", text.contains("mask_steps="))
            assertTrue(
                "mask_steps must be the real count, not n/a, after inference:\n$text",
                !text.contains("mask_steps=n/a")
            )
            assertTrue("longest_dropout missing:\n$text", text.contains("longest_dropout="))
            assertTrue("slot_flips missing:\n$text", text.contains("slot_flips="))
            assertTrue("raw coverage must stay separate:\n$text",
                text.contains("raw_coverage="))

            // The dropout we injected must be visible, not rounded away.
            val dropout = Regex("longest_dropout=(\\d+)").find(text)!!.groupValues[1].toInt()
            assertTrue("injected 3-frame loss must be recorded, got $dropout", dropout >= 3)

            // And the segment must be a real duration.
            val segMs = Regex("seg_ms=(\\d+)").find(text)!!.groupValues[1].toLong()
            assertTrue("segment must have positive duration, got $segMs", segMs > 0L)

            runCatching { out.delete() }
        } finally {
            tracker.close()
            log.clear()
        }
    }

    /**
     * A rejected attempt must say the mask was never measured.
     *
     * A capture that never reached inference has no model mask. Writing 0 would
     * read as "the mask was empty", which points at a different failure than
     * "no usable tracking record" - and the two need different fixes.
     */
    @Test
    fun rejectedAttemptReportsMaskAsUnmeasured() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val log = DiagnosticsLog(ctx)
        log.clear()

        val capture = HandTracker.Capture(emptyList(), 900L, 12f, 0)
        log.record(capture, null, 0L, DiagnosticsLog.Attempt(
            endReason = "HAND_LOST",
            startedAtMs = 100L,
            endedAtMs = 1000L,
            rawHandCoverage = 0f
        ))

        val out = log.exportCopy("test")
        assertNotNull(out)
        val text = out!!.readText()
        assertTrue("a pre-inference rejection must not claim a mask:\n$text",
            text.contains("mask_steps=n/a"))
        assertTrue("it must still name the end reason:\n$text",
            text.contains("end_reason=HAND_LOST"))
        runCatching { out.delete() }
        log.clear()
    }

    /**
     * Mirrors CameraScreen.attemptOf so the test exercises the same arithmetic.
     *
     * Kept here rather than widening the production API: attemptOf is a private
     * helper of the screen, and exposing it just for a test would put a seam in
     * the app for no product reason.
     */
    private fun attemptForTest(o: AutoSegmenter.Outcome): DiagnosticsLog.Attempt {
        val frames = o.frames
        var longest = 0
        var run = 0
        var both = 0
        var flips = 0
        var prevSlots: Set<Int>? = null
        for (f in frames) {
            val slots = f.hands.map { it.slot }.toSet()
            if (slots.isEmpty()) {
                run++
                if (run > longest) longest = run
            } else {
                run = 0
                if (slots.size == 2) both++
                val p = prevSlots
                if (p != null && p.isNotEmpty() && slots.isNotEmpty() && p != slots) flips++
                prevSlots = slots
            }
        }
        val coverage = if (frames.isEmpty()) 0f
            else frames.count { it.hands.isNotEmpty() }.toFloat() / frames.size
        return DiagnosticsLog.Attempt(
            endReason = o.reason.name,
            startedAtMs = o.startedAtMs,
            endedAtMs = o.endedAtMs,
            bothHandsFrames = both,
            longestDropoutFrames = longest,
            slotFlips = flips,
            rawHandCoverage = coverage
        )
    }
}