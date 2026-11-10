package org.echogesture.finalapp

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Owns MediaPipe HandLandmarker (VIDEO mode, 2 hands, thresholds 0.3 — matching
 * src/landmark_frontend.py) and the bounded capture ring. Frames are converted
 * to RGB bitmaps on a background executor; nothing is persisted or uploaded.
 */
class HandTracker private constructor(
    private val landmarker: HandLandmarker
) : AutoCloseable {

    private val nextFrameTs = AtomicLong(0L)
    private val closed = AtomicBoolean(false)
    private val ring = ArrayDeque<Frame>()
    private val lock = Any()

    /** One analysed frame: RGB bitmap + canonical hand slots with presence. */
    data class Hand(val slot: Int, val xyz: FloatArray)
    data class Frame(val bitmap: Bitmap, val hands: List<Hand>)

    companion object {
        private const val FRAME_INTERVAL_MS = 33L // ~30 fps, monotonic VIDEO timestamps

        fun create(context: Context, modelAssetPath: String): HandTracker {
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder().setModelAssetPath(modelAssetPath).build()
                )
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.3f)
                .setMinHandPresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .build()
            return HandTracker(HandLandmarker.createFromOptions(context, options))
        }
    }

    /**
     * Analyses one camera frame. Returns true if the frame was appended, false
     * if the tracker is closed. Safe to call from the ImageAnalysis executor.
     */
    fun analyse(imageProxy: ImageProxy): Boolean {
        if (closed.get()) {
            imageProxy.close()
            return false
        }
        return try {
            val bitmap = imageProxy.toBitmap()
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = landmarker.detectForVideo(
                mpImage,
                ImageProcessingOptions.builder()
                    .setRotationDegrees(imageProxy.imageInfo.rotationDegrees)
                    .build(),
                nextFrameTs.getAndAdd(FRAME_INTERVAL_MS)
            )
            val frame = Frame(bitmap, toHands(result))
            synchronized(lock) {
                ring.addLast(frame)
                while (ring.size > 32) ring.removeFirst()
            }
            mpImage.close()
            true
        } catch (t: Throwable) {
            // A single bad frame must not stop the preview.
            false
        } finally {
            imageProxy.close()
        }
    }

    fun renderLandmarks(frame: Frame, target: Bitmap): Bitmap {
        val canvas = android.graphics.Canvas(target)
        val paint = android.graphics.Paint().apply {
            color = 0xFF00E676.toInt()
            strokeWidth = 6f
            style = android.graphics.Paint.Style.STROKE
            isAntiAlias = true
        }
        val w = frame.bitmap.width.toFloat()
        val h = frame.bitmap.height.toFloat()
        for (hand in frame.hands) {
            val pts = hand.xyz
            for (i in 0 until 21) {
                canvas.drawCircle(pts[i * 3] * w, pts[i * 3 + 1] * h, 4f, paint)
            }
        }
        return target
    }

    /** Snapshot of the current capture window, oldest first. Empty if <1. */
    fun snapshot(): List<Frame> = synchronized(lock) { ring.toList() }

    /** (frameCount, handFrameCount) from the current ring. */
    fun status(): Pair<Int, Int> = synchronized(lock) {
        ring.size to ring.count { it.hands.isNotEmpty() }
    }

    fun clear() = synchronized(lock) { ring.clear() }

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            landmarker.close()
            synchronized(lock) { ring.clear() }
        }
    }

    private fun toHands(result: HandLandmarkerResult): List<Hand> {
        val out = ArrayList<Hand>(result.landmarks().size)
        // handedness() rows align 1:1 with landmarks() rows; take category 0's name.
        var i = 0
        for (pts in result.landmarks()) {
            val handedness = if (i < result.handedness().size && result.handedness()[i].isNotEmpty())
                result.handedness()[i][0].categoryName()
            else "Right"
            // MediaPipe camera-POV handedness -> canonical [Left=0, Right=1],
            // passed straight through (no mirroring; the label is authoritative).
            val slot = if (handedness.equals("Left", ignoreCase = true)) 0 else 1
            val arr = FloatArray(21 * 3)
            for (j in 0 until 21) {
                arr[j * 3] = pts[j].x()
                arr[j * 3 + 1] = pts[j].y()
                arr[j * 3 + 2] = pts[j].z()
            }
            out.add(Hand(slot, arr))
            i++
        }
        return out
    }
}