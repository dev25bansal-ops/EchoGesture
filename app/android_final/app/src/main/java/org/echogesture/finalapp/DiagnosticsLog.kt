package org.echogesture.finalapp

import android.content.Context
import android.util.AtomicFile
import java.io.File
import java.util.Locale

/**
 * Opt-in, on-device record of what each capture actually contained.
 *
 * **What is written:** per-capture frame count, capture duration in real
 * milliseconds, effective analysis FPS, how many frames had a hand, which
 * slots were present, the predicted label, its score, the top-5 and the
 * inference time.
 *
 * **What is never written:** images, video, raw or normalised landmarks, the
 * landmark arrays in any form, the signer's identity, and anything derived
 * from the pixels themselves. Nothing leaves the device — the app has no
 * internet permission at all, so this file cannot be uploaded even in
 * principle.
 *
 * The point is to separate the four failure modes that look identical in a
 * screenshot: too short a capture, the hand leaving the frame, a feature-
 * construction mismatch, and a model that simply misclassifies correct input.
 * Frame counts and FPS answer the first, the presence mask answers the second,
 * and replaying the same clip through both paths answers the rest.
 *
 * Off by default. A consenting test session turns it on explicitly, and
 * [clear] removes the file.
 */
class DiagnosticsLog(context: Context, private val enabled: () -> Boolean = { true }) {

    private val file = File(context.filesDir, FILE_NAME)
    private val storage = AtomicFile(file)
    private val export = File(context.filesDir, "export/echogesture-diagnostics.txt")
    private val lock = Any()

    /**
     * Appends one capture record. Bounded to [MAX_RECORDS] lines so a long
     * session cannot fill the disk; oldest records are dropped first.
     */
    /**
     * Extra per-attempt facts the failure triage needs.
     *
     * Kept separate from [HandTracker.Capture] because these describe the
     * SEGMENTER's decision, not the sampled window: when it decided the sign
     * began and ended, and why it stopped. Without them a wrong prediction
     * cannot be told apart from a correctly-classified fragment of a sign.
     */
    data class Attempt(
        val endReason: String = "",
        val startedAtMs: Long = 0L,
        val endedAtMs: Long = 0L,
        /**
         * Steps present in the 24-step mask ACTUALLY fed to the ensemble, or -1
         * when the attempt never reached inference.
         *
         * Deliberately not derived from hand-bearing frame counts. The mask is
         * built after resampling, so a step can be marked present from
         * interpolated frames; an estimate from raw coverage would report a
         * different number and hide that. -1 is used rather than 0 so a rejected
         * attempt is not mistaken for "the mask was empty".
         */
        val maskSteps: Int = -1,
        /** Frames where both slots were occupied. */
        val bothHandsFrames: Int = 0,
        /** Longest run of consecutive hand-less frames. */
        val longestDropoutFrames: Int = 0,
        /** Times the left/right slot assignment flipped between adjacent frames. */
        val slotFlips: Int = 0,
        /** Raw hand-bearing coverage of the sampled window, 0..1. */
        val rawHandCoverage: Float = 0f
    ) {
        /** This attempt with mask coverage taken from the real model input. */
        fun withMaskFrom(mask: FloatArray): Attempt =
            copy(maskSteps = mask.count { it > 0.5f })
    }

    fun record(
        capture: HandTracker.Capture,
        result: LandmarkClassifier.Result?,
        totalMs: Long,
        attempt: Attempt? = null,
        fromClip: Boolean = false
    ) = synchronized(lock) {
        val present = capture.frames.joinToString("") { f ->
            when (f.hands.map { it.slot }.toSet()) {
                setOf(0) -> "L"
                setOf(1) -> "R"
                setOf(0, 1) -> "B"
                else -> "-"
            }
        }
        val top5 = result?.top5?.joinToString("|") ?: ""
        val score = result?.confidence?.let { String.format(Locale.US, "%.4f", it) } ?: ""
        val line = buildString {
            append("frames=").append(capture.frameCount)
            append(" dur_ms=").append(capture.durationMs)
            append(" fps=").append(String.format(Locale.US, "%.1f", capture.fps))
            append(" hand_frames=").append(capture.handFrames)
            append(" presence=").append(present)
            append(" pred=").append(result?.prediction ?: "")
            append(" score=").append(score)
            append(" top5=").append(top5)
            append(" margin=").append(result?.margin?.let { String.format(Locale.US, "%.4f", it) } ?: "")
            append(" infer_ms=").append(result?.inferenceMillis ?: -1)
            append(" total_ms=").append(totalMs)
            append(" source=").append(if (fromClip) "clip" else "camera")
            if (attempt != null) {
                // end_reason is the segmenter's own verdict. A rejection that
                // happened BEFORE inference shows mask_steps=n/a, so "no usable
                // tracking record" is never read as "the model classified this
                // and got it wrong" - two different failures with different fixes.
                append(" end_reason=").append(attempt.endReason)
                append(" onset_ms=").append(attempt.startedAtMs)
                append(" end_ms=").append(attempt.endedAtMs)
                append(" seg_ms=")
                    .append((attempt.endedAtMs - attempt.startedAtMs).coerceAtLeast(0))
                append(" mask_steps=")
                    .append(if (attempt.maskSteps < 0) "n/a" else "${attempt.maskSteps}/24")
                append(" raw_coverage=").append(String.format(Locale.US, "%.2f", attempt.rawHandCoverage))
                append(" both_hands=").append(attempt.bothHandsFrames)
                append(" longest_dropout=").append(attempt.longestDropoutFrames)
                append(" slot_flips=").append(attempt.slotFlips)
            }
        }

        appendLine(line)
    }

    /**
     * Appends a free-form note from the capture path.
     *
     * Used where a measurement could not be taken, so the absence is recorded
     * as a fact instead of leaving a gap that reads as "nothing to report". A
     * batch of `profile_unavailable` lines is itself the finding: it means the
     * tracker was gone by the time the capture was logged.
     */
    fun recordNote(note: String) = synchronized(lock) {
        appendLine("note $note")
    }

    fun read(): String = synchronized(lock) {
        readRecord()
    }

    /**
     * Appends the measured analysis-path profile.
     *
     * This is the record that answers "how fast is the analyser actually
     * running on this phone" with numbers instead of speculation: the stage
     * split says whether the detector or the YUV-to-Bitmap copy dominates, the
     * arrival interval says what the camera delivered, and the derived drop
     * count says how many frames the analyser never saw.
     *
     * Deliberately separate from the per-capture records above, because it
     * describes the camera, not a sign. Reading an analysis rate off a capture's
     * frame count is what produced the old "5-10 fps" figure from a run that was
     * actually capped at 32 samples.
     *
     * No pixels, no landmarks, no identifiers.
     */
    fun recordProfiler(snapshot: AnalysisProfiler.Snapshot) = synchronized(lock) {
        val line = buildString {
            append("profile fps=").append(String.format(Locale.US, "%.1f", snapshot.effectiveFps))
            append(" interval_p50=").append(snapshot.intervalP50)
            append(" interval_p95=").append(snapshot.intervalP95)
            append(" analyse_p50=").append(snapshot.analysisP50)
            append(" convert_p50=").append(snapshot.convertP50)
            append(" detect_p50=").append(snapshot.detectP50)
            append(" detect_p95=").append(snapshot.detectP95)
            append(" delivered=").append(snapshot.delivered)
            append(" dropped=").append(snapshot.dropped)
            append(" limit=").append(snapshot.limit)
        }

        appendLine(line)
    }

    private fun readRecord(): String = if (enabled()) {
        runCatching { storage.readFully().toString(Charsets.UTF_8) }.getOrDefault("")
    } else ""

    private fun appendLine(line: String) {
        if (!enabled()) return
        val existing = readRecord().lineSequence().filter { it.isNotBlank() && !it.startsWith("#") }
        val kept = (existing.toList() + line).takeLast(MAX_RECORDS)
        val stream = storage.startWrite()
        try {
            stream.write((HEADER + "\n" + kept.joinToString("\n") + "\n").toByteArray(Charsets.UTF_8))
            storage.finishWrite(stream)
        } catch (exception: Exception) {
            storage.failWrite(stream)
            throw exception
        }
    }

    fun clear() = synchronized(lock) {
        storage.delete()
        export.delete()
        Unit
    }

    fun exists(): Boolean = synchronized(lock) { enabled() && file.isFile }

    /**
     * Copies the record into a shareable file and returns it, or null when
     * there is nothing to share.
     *
     * Written to `files/export/`, which is the only directory the FileProvider
     * in `res/xml/file_paths.xml` exposes. The copy is separate from the live
     * record so a share can never be handed a file the app keeps appending to
     * while another process reads it.
     *
     * Called only from an explicit user tap.
     */
    fun exportCopy(generatedAt: String): java.io.File? = synchronized(lock) {
        val body = readRecord()
        if (body.lineSequence().none { it.startsWith("frames=") }) return null
        runCatching {
            export.parentFile?.mkdirs()
            export.writeText(
                "# Exported from EchoGesture settings on $generatedAt.\n" +
                    "# Capture statistics only: frame counts, timings, presence\n" +
                    "# flags and predicted glosses. No images, no landmarks, no\n" +
                    "# identifiers. Shared only because the user chose to.\n" +
                    body
            )
            export
        }.getOrNull()
    }

    /** Path shown to the user so they can retrieve it over USB adb. */
    fun path(): String = file.absolutePath

    companion object {
        const val FILE_NAME = "diagnostics.txt"
        private const val MAX_RECORDS = 500
        private const val HEADER =
            "# EchoGesture capture diagnostics. No images, no landmarks, no identifiers."
    }
}
