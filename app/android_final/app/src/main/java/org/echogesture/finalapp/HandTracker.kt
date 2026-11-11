package org.echogesture.finalapp

import android.content.Context
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
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

    /** Consecutive failed frames; reset to 0 on any successful analysis. */
    private val consecutiveFailures = AtomicInteger(0)

    /**
     * One analysed frame: the canonical [Left=0, Right=1] hand slots for that
     * instant. Frames with no detection carry an empty list so their temporal
     * position is preserved (matching src/landmark_frontend.py, which appends a
     * zero-landmark, zero-presence entry for every frame). The source bitmap is
     * deliberately not retained: nothing after tracking needs it, and holding
     * 32 ARGB_8888 buffers would be tens of MB.
     */
    data class Hand(val slot: Int, val xyz: FloatArray)
    data class Frame(val hands: List<Hand>)

    companion object {
        private const val TAG = "HandTracker"
        private const val FRAME_INTERVAL_MS = 33L // ~30 fps, monotonic VIDEO timestamps

        /** Capture window. Matches src/landmark_frontend.py's max_frames=32. */
        const val RING_CAPACITY = 32

        /**
         * Consecutive frame failures after which the tracker is considered
         * persistently broken (not a one-off bad frame). At ~30 fps this is
         * ~1 s of continuous failure. Surfaced to the UI so a dead detector is
         * reported rather than silently showing an empty preview forever.
         */
        const val FAILURE_THRESHOLD = 30

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
     * otherwise. [ImageProxy.close] is guaranteed exactly once on every path,
     * including a closed tracker. Safe to call from the ImageAnalysis executor.
     */
    fun analyse(imageProxy: ImageProxy): Boolean {
        if (closed.get()) {
            imageProxy.close()
            return false
        }
        var mpImage: MPImage? = null
        return try {
            // toBitmap() copies the YUV planes into a fresh ARGB_8888 bitmap
            // and does NOT apply rotation; rotation is passed to MediaPipe via
            // ImageProcessingOptions below, exactly once.
            val bitmap = imageProxy.toBitmap()
            mpImage = BitmapImageBuilder(bitmap).build()
            // detectForVideo() is synchronous: by the time it returns MediaPipe
            // has copied the pixels into its packet, so closing mpImage (which
            // recycles `bitmap`) afterwards is safe.
            val result = landmarker.detectForVideo(
                mpImage,
                ImageProcessingOptions.builder()
                    .setRotationDegrees(imageProxy.imageInfo.rotationDegrees)
                    .build(),
                nextFrameTs.getAndAdd(FRAME_INTERVAL_MS)
            )
            val frame = Frame(toHands(result))
            synchronized(lock) {
                ring.addLast(frame)
                while (ring.size > RING_CAPACITY) ring.removeFirst()
            }
            consecutiveFailures.set(0)
            true
        } catch (t: Throwable) {
            // A single bad frame must not stop the preview, but a run of them
            // means the detector is broken. Count consecutive failures so the
            // UI can surface a real error instead of a silently empty preview.
            // Only the exception type is logged — never frame or image data.
            consecutiveFailures.incrementAndGet()
            Log.w(TAG, "Hand analysis failed (${t.javaClass.simpleName})")
            false
        } finally {
            mpImage?.close()
            imageProxy.close()
        }
    }

    /** Consecutive failed frames since the last successful analysis. */
    fun consecutiveFailures(): Int = consecutiveFailures.get()

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