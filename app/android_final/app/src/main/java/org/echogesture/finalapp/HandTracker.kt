package org.echogesture.finalapp

import android.content.Context
import android.os.SystemClock
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

    /**
     * A capture collects exactly [RING_CAPACITY] frames starting from the
     * moment the user asks for one. Without this the ring would already be
     * full of frames from before the user raised their hand, so the 24-frame
     * sequence the model sees would mix an old window with the new sign.
     */
    private var capturing = false
    private var captureStartedMs = 0L
    private var captureCallback: (() -> Unit)? = null

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
    data class Frame(val hands: List<Hand>, val capturedAtMs: Long)

    /** What one finished capture actually contained, for the UI and diagnostics. */
    data class Capture(
        val frames: List<Frame>,
        val durationMs: Long,
        val fps: Float,
        val handFrames: Int
    ) {
        val frameCount: Int get() = frames.size
    }

    companion object {
        private const val TAG = "HandTracker"
        private const val FRAME_INTERVAL_MS = 33L // ~30 fps, monotonic VIDEO timestamps

        /** Capture window. Matches src/landmark_frontend.py's max_frames=32. */
        const val RING_CAPACITY = 32

        /**
         * Initial window length. The window is 32 frames, and at 30 fps that is
         * ~1.07 s — but real analysis FPS is lower than the camera's frame rate
         * because hand tracking runs inline, so the window is longer than this
         * in wall clock. Training sampled 32 frames spread across each whole
         * ASL Citizen clip (one example was 110 frames), so a live window that
         * is too short only sees the part of the sign that happens to fall
         * inside it.
         *
         * The window therefore extends until the capture covers
         * [MIN_CAPTURE_MS] of real time, using measured frame timestamps, with
         * [RING_CAPACITY] as the floor. It stays bounded so capture cannot run
         * away when the analyser stalls.
         */
        const val MIN_CAPTURE_MS = 1000L
        const val MAX_CAPTURE_MS = 4000L
        const val MAX_CAPTURE_FRAMES = 96

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
            val now = SystemClock.elapsedRealtime()
            val frame = Frame(toHands(result), now)
            var completed: (() -> Unit)? = null
            synchronized(lock) {
                ring.addLast(frame)
                while (ring.size > RING_CAPACITY) ring.removeFirst()
                // End the capture once it covers the minimum real-time window,
                // with RING_CAPACITY as the floor and MAX_CAPTURE_* as a bound.
                // A stalled analyser therefore cannot keep it open forever.
                val elapsed = now - captureStartedMs
                val enoughTime = elapsed >= MIN_CAPTURE_MS
                val tooLong = elapsed >= MAX_CAPTURE_MS || ring.size >= MAX_CAPTURE_FRAMES
                if (capturing && ((ring.size >= RING_CAPACITY && enoughTime) || tooLong)) {
                    capturing = false
                    completed = captureCallback
                    captureCallback = null
                }
            }
            // Invoked outside the lock: the callback hands the finished window
            // to the inference executor and must not hold the ring's monitor.
            completed?.invoke()
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

    /**
     * Starts a fresh capture window: clears the ring and calls [onComplete]
     * once [RING_CAPACITY] new frames have been analysed. Any previous
     * capture in flight is cancelled. Returns immediately; the callback runs
     * on the analysis thread.
     */
    fun beginCapture(onComplete: () -> Unit) {
        synchronized(lock) {
            ring.clear()
            captureStartedMs = SystemClock.elapsedRealtime()
            captureCallback = onComplete
            capturing = true
        }
    }

    fun cancelCapture() = synchronized(lock) {
        capturing = false
        captureCallback = null
    }

    /** True while a capture window is still filling. */
    fun isCapturing(): Boolean = synchronized(lock) { capturing }

    /**
     * The finished capture window, with the real elapsed time and effective
     * analysis FPS measured from per-frame timestamps.
     *
     * The FPS here is what the analyser actually achieved, not the camera's
     * nominal rate: hand tracking runs inline, and a slower device produces a
     * longer window for the same 32 frames. Reporting the nominal 30 fps
     * would misstate both the capture duration and the temporal span the model
     * saw.
     */
    fun captureSnapshot(): Capture = synchronized(lock) {
        val frames = ring.toList()
        val first = frames.firstOrNull()?.capturedAtMs ?: 0L
        val last = frames.lastOrNull()?.capturedAtMs ?: 0L
        val duration = if (frames.size >= 2) last - first else 0L
        val fps = if (duration > 0) (frames.size - 1) * 1000f / duration else 0f
        Capture(frames, duration, fps, frames.count { it.hands.isNotEmpty() })
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            landmarker.close()
            synchronized(lock) {
                ring.clear()
                capturing = false
                captureCallback = null
            }
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