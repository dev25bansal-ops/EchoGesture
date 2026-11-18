package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sin

/**
 * The auto-segmenter's contract, exercised with synthetic landmark sequences.
 *
 * These tests are the reason the feature can be claimed at all. "It auto-stops
 * on a real sign" is not verifiable from one recording; each threshold's
 * behaviour under a specific adversarial sequence is.
 */
class AutoSegmenterTest {

    private val cfg = AutoSegmenter.Config()

    /** A hand at [x],[y] with a small wrist offset, so it is a plausible pose. */
    private fun handAt(slot: Int, x: Float, y: Float, z: Float = 0f): HandTracker.Hand {
        val xyz = FloatArray(21 * 3)
        for (i in 0 until 21) {
            xyz[i * 3] = x + i * 0.001f
            xyz[i * 3 + 1] = y + i * 0.001f
            xyz[i * 3 + 2] = z
        }
        return HandTracker.Hand(slot, xyz)
    }

    private fun frame(at: Long, vararg hands: HandTracker.Hand) =
        HandTracker.Frame(hands.toList(), at)

    /** Feeds frames at [fps], calling [pose] with the frame index. */
    private fun run(
        seg: AutoSegmenter,
        startMs: Long,
        count: Int,
        fps: Float,
        pose: (Int) -> FloatArray
    ): AutoSegmenter.Outcome? {
        val step = (1000f / fps).toLong()
        for (i in 0 until count) {
            val p = pose(i)
            val f = if (p[0] < 0f) frame(startMs + i * step)
            else frame(startMs + i * step, handAt(0, p[0], p[1], p.getOrElse(2) { 0f }))
            val out = seg.offer(f)
            if (out != null) return out
        }
        return null
    }

    // ---- onset -------------------------------------------------------------

    @Test
    fun `does not arm a sign from frames before the tap`() {
        val seg = AutoSegmenter(cfg)
        // Motion arrives, but the user has not tapped Start.
        run(seg, 0, 40, 10f) { i -> floatArrayOf(0.10f + i * 0.02f, 0.5f) }
        assertEquals(AutoSegmenter.State.IDLE, seg.state)
        assertTrue("nothing may be buffered while idle", seg.frameCount() == 0)
    }

    @Test
    fun `a lone detection during acquisition does not start a sign`() {
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        // One frame with a hand, then nothing: must not be read as a sign.
        seg.offer(frame(40, handAt(0, 0.5f, 0.5f)))
        for (i in 1..20) seg.offer(frame(40L + i * 100, ))
        assertEquals(AutoSegmenter.State.ARMED, seg.state)
    }

    @Test
    fun `sustained motion starts a sign`() {
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        // Move at a steady rate that stays above the frozen floor, and confirm
        // the state advances without the sign being ended early by a floor that
        // drifted up to meet the motion.
        for (i in 0 until 60) {
            val t = 100L + i * 100L
            val f = if (i < 5) frame(t)
            else frame(t, handAt(0, 0.2f + i * 0.03f, 0.5f))
            val out = seg.offer(f)
            assertNull("a steadily moving sign must not end at frame $i", out)
        }
        assertEquals(AutoSegmenter.State.SIGNING, seg.state)
        assertTrue("pre-roll must contribute frames", seg.frameCount() > 0)
    }

    @Test
    fun `a held static pose also starts a sign`() {
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        // Hand appears and does not move at all — a held handshape. Frames are
        // 100 ms apart, so the 450 ms pose-hold threshold is reached well
        // inside this run.
        var armed = false
        var out: AutoSegmenter.Outcome? = null
        for (i in 0 until 20) {
            if (seg.state == AutoSegmenter.State.SIGNING) armed = true
            out = seg.offer(frame(100L + i * 100L, handAt(0, 0.5f, 0.5f))) ?: out
        }
        assertTrue(
            "a static sign must arm, or held-handshape signs can never be read",
            armed
        )
        // A hand that never moves also never gives a completion cue, so the
        // quiet rule ends it. That is the intended outcome, not a failure.
        assertNotNull("a held pose must eventually auto-stop", out)
        assertEquals(AutoSegmenter.EndReason.QUIET, out!!.reason)
    }

    @Test
    fun `a static sign still yields a usable segment`() {
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        var out: AutoSegmenter.Outcome? = null
        for (i in 0 until 40) {
            out = seg.offer(frame(100L + i * 100L, handAt(0, 0.5f, 0.5f))) ?: out
            if (out != null) break
        }
        assertNotNull(out)
        assertTrue(
            "a held handshape must be long enough to classify, got ${out!!.durationMs}ms",
            out.isUsable
        )
        assertTrue("segment must carry the sign frames", out.frames.isNotEmpty())
    }

    @Test
    fun `a sign does not start before the arming delay`() {
        val seg = AutoSegmenter(cfg.copy(armingMs = 500L))
        seg.arm(0)
        // Fast motion immediately after arming.
        seg.offer(frame(50, handAt(0, 0.1f, 0.5f)))
        seg.offer(frame(100, handAt(0, 0.4f, 0.5f)))
        seconds(seg, 120, 10)
        assertEquals(AutoSegmenter.State.ARMED, seg.state)
    }

    // ---- completion --------------------------------------------------------

    @Test
    fun `a completed sign auto-stops after the quiet period`() {
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        // Settle, sign for ~1.2 s, then hold still.
        var i = 0
        fun step(): AutoSegmenter.Outcome? {
            val t = i * 100L
            val f = when {
                i < 5 -> frame(t)
                i < 17 -> frame(t, handAt(0, 0.2f + (i - 5) * 0.04f, 0.5f))
                else -> frame(t, handAt(0, 0.68f, 0.5f))
            }
            i++
            return seg.offer(f)
        }
        var out: AutoSegmenter.Outcome? = null
        var n = 0
        while (out == null && n < 200) { out = step(); n++ }
        assertNotNull("a finished sign must auto-stop", out)
        assertEquals(AutoSegmenter.EndReason.QUIET, out!!.reason)
        assertTrue("segment must be usable, was ${out.reason}", out.isUsable)
        assertTrue("segment too short: ${out.durationMs}ms", out.durationMs >= cfg.minSignMs)
        assertEquals(AutoSegmenter.State.IDLE, seg.state)
    }

    @Test
    fun `a pause inside a two-part sign does not stop it`() {
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        var i = 0
        fun step(): AutoSegmenter.Outcome? {
            val t = i * 100L
            val f = when {
                i < 5 -> frame(t)
                // part 1: moving
                i < 15 -> frame(t, handAt(0, 0.2f + (i - 5) * 0.04f, 0.5f))
                // internal pause, shorter than quietHoldMs (800 ms)
                i < 21 -> frame(t, handAt(0, 0.6f, 0.5f))
                // part 2: moving again
                i < 31 -> frame(t, handAt(0, 0.6f - (i - 21) * 0.04f, 0.5f))
                else -> frame(t, handAt(0, 0.2f, 0.5f))
            }
            i++
            return seg.offer(f)
        }
        var out: AutoSegmenter.Outcome? = null
        var n = 0
        // The pause is 600 ms; the sign must still be running at its end.
        while (n < 26) { out = step(); n++ }
        assertNull("a 600 ms internal pause must not end the sign", out)
        assertEquals(AutoSegmenter.State.SIGNING, seg.state)
    }

    @Test
    fun `the sign eventually stops after a long pause`() {
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        val out = run(seg, 0, 120, 10f) { i ->
            when {
                i < 5 -> floatArrayOf(-1f)
                i < 20 -> floatArrayOf(0.2f + i * 0.03f, 0.5f)
                else -> floatArrayOf(0.8f, 0.5f)
            }
        }
        assertNotNull(out)
        assertEquals(AutoSegmenter.EndReason.QUIET, out!!.reason)
    }

    @Test
    fun `a sign that never goes quiet times out`() {
        val seg = AutoSegmenter(cfg.copy(maxSignMs = 2_000L))
        seg.arm(0)
        val out = run(seg, 0, 200, 10f) { i ->
            if (i < 5) floatArrayOf(-1f)
            // Constant motion forever.
            else floatArrayOf(0.3f + 0.15f * sin(i / 2.0).toFloat(), 0.5f)
        }
        assertNotNull(out)
        assertEquals(AutoSegmenter.EndReason.TIMEOUT, out!!.reason)
    }

    @Test
    fun `a hand leaving the frame ends an interrupted sign`() {
        val seg = AutoSegmenter(cfg.copy(handLossGraceMs = 400L))
        seg.arm(0)
        val out = run(seg, 0, 80, 10f) { i ->
            when {
                i < 5 -> floatArrayOf(-1f)
                i < 25 -> floatArrayOf(0.2f + i * 0.03f, 0.5f)
                else -> floatArrayOf(-1f) // hand gone
            }
        }
        assertNotNull(out)
        assertEquals(AutoSegmenter.EndReason.HAND_LOST, out!!.reason)
    }

    // ---- boundaries --------------------------------------------------------

    @Test
    fun `consecutive signs do not share frames`() {
        val seg = AutoSegmenter(cfg)

        val first = captureOneSign(seg, startMs = 0)
        assertNotNull(first)
        val second = captureOneSign(seg, startMs = 10_000)
        assertNotNull(second)

        val firstEnd = first!!.frames.maxOf { it.capturedAtMs }
        val secondStart = second!!.frames.minOf { it.capturedAtMs }
        assertTrue(
            "second sign reused frames from the first ($firstEnd >= $secondStart)",
            secondStart > firstEnd
        )
    }

    @Test
    fun `slow 5 fps input still segments without a frame-count trigger`() {
        // The phone analyser runs at 5-10 fps. A rule expressed in frames would
        // behave differently here than at 30 fps; every threshold is in ms.
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        val out = run(seg, 0, 60, 5f) { i ->
            when {
                i < 4 -> floatArrayOf(-1f)
                i < 14 -> floatArrayOf(0.2f + i * 0.05f, 0.5f)
                else -> floatArrayOf(0.9f, 0.5f)
            }
        }
        assertNotNull("5 fps input must still auto-stop", out)
        assertEquals(AutoSegmenter.EndReason.QUIET, out!!.reason)
    }

    @Test
    fun `manual finish works even while a sign is running`() {
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        run(seg, 0, 20, 10f) { i ->
            if (i < 5) floatArrayOf(-1f) else floatArrayOf(0.2f + i * 0.03f, 0.5f)
        }
        assertEquals(AutoSegmenter.State.SIGNING, seg.state)
        val out = seg.finishManually(2_000)
        assertEquals(AutoSegmenter.EndReason.MANUAL, out.reason)
        assertTrue(out.frames.isNotEmpty())
    }

    @Test
    fun `manual finish on a fragment is reported as too short`() {
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        // Only a few frames in, before the segmenter has confirmed any sign.
        // Finish must not turn that into a classification.
        seg.offer(frame(100, handAt(0, 0.5f, 0.5f)))
        seg.offer(frame(200, handAt(0, 0.52f, 0.5f)))
        val out = seg.finishManually(300)
        assertEquals(
            "a fragment must be rejected, not classified",
            AutoSegmenter.EndReason.TOO_SHORT,
            out.reason
        )
        assertTrue(!out.isUsable)
    }

    @Test
    fun `manual finish before onset discards rather than classifying`() {
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        // Armed but nothing has happened yet: no hand has even been seen.
        for (i in 0 until 10) seg.offer(frame(i * 100L))
        val out = seg.finishManually(1_000)
        assertEquals(AutoSegmenter.EndReason.TOO_SHORT, out.reason)
        assertTrue("no frames means nothing to classify", out.frames.isEmpty())
    }

    @Test
    fun `cancel discards the attempt`() {
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        run(seg, 0, 20, 10f) { i ->
            if (i < 5) floatArrayOf(-1f) else floatArrayOf(0.2f + i * 0.03f, 0.5f)
        }
        seg.cancel()
        assertEquals(AutoSegmenter.State.IDLE, seg.state)
        assertEquals(0, seg.frameCount())
    }

    @Test
    fun `arming twice does not splice two attempts`() {
        val seg = AutoSegmenter(cfg)
        seg.arm(0)
        run(seg, 0, 20, 10f) { i ->
            if (i < 5) floatArrayOf(-1f) else floatArrayOf(0.2f + i * 0.03f, 0.5f)
        }
        val before = seg.frameCount()
        assertTrue(before > 0)
        seg.arm(5_000)
        assertEquals("a re-arm must start empty", 0, seg.frameCount())
        assertEquals(AutoSegmenter.State.ARMED, seg.state)
    }

    /** Runs one full arm → sign → quiet cycle and returns its outcome. */
    private fun captureOneSign(seg: AutoSegmenter, startMs: Long): AutoSegmenter.Outcome? {
        seg.arm(startMs)
        val out = run(seg, startMs, 120, 10f) { i ->
            when {
                i < 5 -> floatArrayOf(-1f)
                i < 18 -> floatArrayOf(0.2f + i * 0.03f, 0.5f)
                else -> floatArrayOf(0.7f, 0.5f)
            }
        }
        return out
    }

    private fun seconds(seg: AutoSegmenter, fromMs: Long, n: Int) {
        for (i in 0 until n * 10) seg.offer(frame(fromMs + i * 100L))
    }
}
