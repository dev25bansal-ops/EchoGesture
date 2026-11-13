package org.echogesture.finalapp

import android.content.Context
import java.io.File

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
class DiagnosticsLog(context: Context) {

    private val file = File(context.filesDir, FILE_NAME)
    private val lock = Any()

    /**
     * Appends one capture record. Bounded to [MAX_RECORDS] lines so a long
     * session cannot fill the disk; oldest records are dropped first.
     */
    fun record(
        capture: HandTracker.Capture,
        result: LandmarkClassifier.Result?,
        totalMs: Long
    ) = synchronized(lock) {
        val present = capture.frames.joinToString("") { f ->
            if (f.hands.isEmpty()) "0" else f.hands.joinToString("") { it.slot.toString() }
        }
        val top5 = result?.top5?.joinToString("|") ?: ""
        val score = result?.confidence?.let { String.format("%.4f", it) } ?: ""
        val line = buildString {
            append("frames=").append(capture.frameCount)
            append(" dur_ms=").append(capture.durationMs)
            append(" fps=").append(String.format("%.1f", capture.fps))
            append(" hand_frames=").append(capture.handFrames)
            append(" presence=").append(present)
            append(" pred=").append(result?.prediction ?: "")
            append(" score=").append(score)
            append(" top5=").append(top5)
            append(" margin=").append(result?.margin?.let { String.format("%.4f", it) } ?: "")
            append(" infer_ms=").append(result?.inferenceMillis ?: -1)
            append(" total_ms=").append(totalMs)
        }

        val existing = runCatching {
            if (file.isFile) file.readLines() else emptyList()
        }.getOrDefault(emptyList()).filter { it.isNotBlank() }

        val kept = (existing + line).takeLast(MAX_RECORDS)
        file.writeText(HEADER + "\n" + kept.joinToString("\n") + "\n")
        Unit
    }

    fun read(): String = synchronized(lock) {
        runCatching { if (file.isFile) file.readText() else "" }.getOrDefault("")
    }

    fun clear() = synchronized(lock) {
        runCatching { file.delete() }
        Unit
    }

    fun exists(): Boolean = file.isFile

    /** Path shown to the user so they can retrieve it over USB adb. */
    fun path(): String = file.absolutePath

    companion object {
        const val FILE_NAME = "diagnostics.txt"
        private const val MAX_RECORDS = 500
        private const val HEADER =
            "# EchoGesture capture diagnostics. No images, no landmarks, no identifiers."
    }
}