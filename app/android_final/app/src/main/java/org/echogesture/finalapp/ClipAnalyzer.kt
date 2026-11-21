package org.echogesture.finalapp

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker

/** Samples the entire selected clip and keeps only landmarks. Runs on the inference executor. */
internal class ClipAnalyzer(private val context: Context) : AutoCloseable {
    private var detector: HandLandmarker? = null

    fun analyze(uri: Uri, keepGoing: () -> Boolean = { true }): SignRecorder.Recording {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()?.takeIf { it > 0 } ?: error("Video duration unavailable")
            val landmarker = detector ?: HandTracker.createLandmarker(
                context, ModelContract.DETECTOR_ASSET, RunningMode.IMAGE
            ).also { detector = it }
            val frames = ArrayList<HandTracker.Frame>(SAMPLES)
            val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                ?.toIntOrNull() ?: 0
            if (rotation == 0) {
                val decoded = SequentialClipDecoder.sample(
                    context, uri, durationMs, SAMPLES, MAX_EDGE, keepGoing
                ) { bitmap, timeMs ->
                    frames.add(track(landmarker, bitmap, timeMs))
                }
                check(keepGoing()) { "Clip analysis cancelled" }
                if (decoded >= SAMPLES * 9 / 10 && frames.last().capturedAtMs >= durationMs * 9 / 10) {
                    Log.i("ClipAnalyzer", "Sampled ${frames.size} frames using sequential decode")
                    return SignRecorder.Recording(frames, 0, durationMs)
                }
                frames.clear()
            }
            for (index in 0 until SAMPLES) {
                check(keepGoing()) { "Clip analysis cancelled" }
                val timestampUs = (durationMs * 1000 - 1) * index / (SAMPLES - 1)
                val bitmap = if (Build.VERSION.SDK_INT >= 27) {
                    retriever.getScaledFrameAtTime(
                        timestampUs, MediaMetadataRetriever.OPTION_CLOSEST, MAX_EDGE, MAX_EDGE
                    )
                } else {
                    retriever.getFrameAtTime(timestampUs, MediaMetadataRetriever.OPTION_CLOSEST)
                } ?: continue
                frames.add(track(landmarker, bitmap, timestampUs / 1000))
            }
            require(frames.size >= 2) { "Video contains fewer than two readable frames" }
            Log.i("ClipAnalyzer", "Sampled ${frames.size} frames using the retriever")
            return SignRecorder.Recording(frames, 0, durationMs)
        } finally {
            retriever.release()
        }
    }

    private fun track(landmarker: HandLandmarker, bitmap: Bitmap, timeMs: Long): HandTracker.Frame {
        val rgb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap else {
            try {
                requireNotNull(bitmap.copy(Bitmap.Config.ARGB_8888, false))
            } finally {
                bitmap.recycle()
            }
        }
        val image = try {
            BitmapImageBuilder(rgb).build()
        } catch (ex: RuntimeException) {
            rgb.recycle()
            throw ex
        }
        return image.use {
            HandTracker.Frame(HandTracker.toHands(landmarker.detect(it)), timeMs)
        }
    }

    override fun close() {
        detector?.close()
        detector = null
    }

    private companion object {
        const val SAMPLES = 32
        const val MAX_EDGE = 640
    }
}
