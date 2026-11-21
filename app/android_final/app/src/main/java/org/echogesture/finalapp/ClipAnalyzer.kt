package org.echogesture.finalapp

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker

/** Samples the entire selected clip and keeps only landmarks. Runs on the inference executor. */
internal class ClipAnalyzer(private val context: Context) : AutoCloseable {
    private var detector: HandLandmarker? = null

    fun analyze(uri: Uri): SignRecorder.Recording {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()?.takeIf { it > 0 } ?: error("Video duration unavailable")
            val landmarker = detector ?: HandTracker.createLandmarker(
                context, ModelContract.DETECTOR_ASSET, RunningMode.IMAGE
            ).also { detector = it }
            val frames = ArrayList<HandTracker.Frame>(32)
            for (index in 0 until 32) {
                val timestampUs = (durationMs * 1000 - 1) * index / 31
                val bitmap = if (Build.VERSION.SDK_INT >= 27) {
                    retriever.getScaledFrameAtTime(
                        timestampUs, MediaMetadataRetriever.OPTION_CLOSEST, 640, 640
                    )
                } else {
                    retriever.getFrameAtTime(timestampUs, MediaMetadataRetriever.OPTION_CLOSEST)
                } ?: continue
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
                try {
                    val hands = HandTracker.toHands(landmarker.detect(image))
                    frames.add(HandTracker.Frame(hands, timestampUs / 1000))
                } finally {
                    image.close()
                }
            }
            require(frames.size >= 2) { "Video contains fewer than two readable frames" }
            return SignRecorder.Recording(frames, 0, durationMs)
        } finally {
            retriever.release()
        }
    }

    override fun close() {
        detector?.close()
        detector = null
    }
}
