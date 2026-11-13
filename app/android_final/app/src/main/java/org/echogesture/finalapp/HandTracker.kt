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
     * A capture collects frames over [CAPTURE_SPAN_MS] starting from the
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

        /** Frames resampled across the window into the model's 24-step input. */
        const val RING_CAPACITY = 32

        /**
         * The capture window is a span of REAL TIME, not a frame count.
         *
         * Training sampled 32 frames spread evenly across each whole ASL
         * Citizen clip (`read_frames()` uses `np.linspace(0, total-1, 32)`); the
         * extractor's own docstring records a typical clip as 110 frames, about
         * 3.7 s. Because `features.py` computes velocity as a plain per-frame
         * difference (`vel[1:] = norm[1:] - norm[:-1]`) with no `dt`
         * normalisation, the step between consecutive sampled frames IS the
         * time base. Across a 3.7 s clip 32 samples are ~115 ms apart; 32
         * consecutive camera frames are ~33 ms apart. Feeding consecutive
         * frames therefore made every velocity ~3.5x too small and every
         * acceleration ~12x too small — the model saw a sliver of the sign
         * played at unit time.
         *
         * So the app now captures a [CAPTURE_SPAN_MS] window and resamples
         * [RING_CAPACITY] frames evenly across it, which reproduces the
         * training step instead of whatever rate the analyser happened to hit.
         * Nothing is invented: every resampled point is an actual analysed
         * frame, chosen by nearest timestamp. If the analyser is too slow to
         * fill the span, the per-frame timestamps below still say what was
         * really covered.
         */
        const val CAPTURE_SPAN_MS = 3700L

        /**
         * Bounds on the raw ring. The span is what matters; these only stop a
         * stalled or runaway analyser from growing memory without limit.
         */
        const val MAX_CAPTURE_FRAMES = 200

        /**
         * Indices of [target] frames spread evenly across the window by capture
         * time — the same `linspace(0, n-1, target)` selection `read_frames()`
         * makes over a training clip. Each entry is a real analysed frame (nearest
         * timestamp), never an interpolated synthetic one. Windows shorter than
         * [target] are returned as they are; the caller resamples them as before.
         *
         * Pure so the selection can be tested without a device: the defect it
         * guards is about which frames are chosen, not about MediaPipe.
         */
        fun evenlySpacedIndices(timestamps: LongArray, target: Int): IntArray {
            if (timestamps.isEmpty()) return IntArray(0)
            if (timestamps.size <= target) return IntArray(timestamps.size) { it }
            val first = timestamps.first()
            val span = (timestamps.last() - first).coerceAtLeast(1L)
            return IntArray(target) { o ->
                val wanted = first + span * o / (target - 1)
                var best = 0
                for (i in timestamps.indices) {
                    if (kotlin.math.abs(timestamps[i] - wanted) < kotlin.math.abs(timestamps[best] - wanted)) best = i
                }
                best
            }
        }

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
        // Timestamp when the frame ARRIVED, not when detection finished.
        // detectForVideo is synchronous and can take tens of milliseconds; a
        // frame that arrived before the user tapped must not be credited to
        // the capture window just because its analysis finished after it.
        val frameAtMs = SystemClock.elapsedRealtime()
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
            val frame = Frame(toHands(result), frameAtMs)
            var completed: (() -> Unit)? = null
            synchronized(lock) {
                // The ring is a rolling buffer of the recent past. It is NOT
                // gated on `capturing`: the sign happens BEFORE the tap, so the
                // frames that describe it are already behind us when the user
                // presses the button. Discarding pre-tap frames here (as an
                // earlier version did) meant the window contained only the
                // seconds AFTER the sign — usually an empty room.
                ring.addLast(frame)
                while (ring.size > MAX_CAPTURE_FRAMES) ring.removeFirst()
                val elapsed = now - captureStartedMs
                // End on time; the frame cap is only a memory backstop for a
                // device so slow that the span barely fills.
                if (capturing && (elapsed >= CAPTURE_SPAN_MS || ring.size >= MAX_CAPTURE_FRAMES)) {
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
     * Classifies the sign that just happened.
     *
     * The signer performs, THEN taps. So the frames describing the sign are
     * already in the rolling ring when this is called, and the window is the
     * [CAPTURE_SPAN_MS] of real time immediately BEFORE the tap — not after it.
     *
     * An earlier version cleared the ring and waited for a fresh window. That
     * required the signer to keep performing after tapping, which the on-screen
     * instruction never asked for: it said "perform one sign, then tap
     * Recognise sign". Measured on a training clip, the hand occupies frames
     * 23–54 of 75; a window starting after the sign contains zero sign frames.
     * That is a capture defect, not a model defect.
     *
     * The callback runs synchronously on the caller's thread if enough history
     * already exists, otherwise after the first frame that completes the window.
     * Returns immediately.
     */
    fun beginCapture(onComplete: () -> Unit) {
        var readyNow = false
        synchronized(lock) {
            captureStartedMs = SystemClock.elapsedRealtime()
            captureCallback = onComplete
            // Enough buffered history to classify right away?
            val first = ring.firstOrNull()?.capturedAtMs
            readyNow = ring.size >= RING_CAPACITY &&
                first != null && (captureStartedMs - first) >= CAPTURE_SPAN_MS * 3 / 4
            capturing = !readyNow
            if (readyNow) captureCallback = null
        }
        if (readyNow) onComplete()
    }

    fun cancelCapture() = synchronized(lock) {
        capturing = false
        captureCallback = null
    }

    /** True while a capture window is still filling. */
    fun isCapturing(): Boolean = synchronized(lock) { capturing }

    /**
     * The window ending now: the most recent [CAPTURE_SPAN_MS] of analysed
     * frames, with real elapsed time and achieved FPS measured from per-frame
     * timestamps.
     *
     * Because the window ends at the tap rather than starting there, it
     * contains the sign the user just performed. [RING_CAPACITY] frames are
     * chosen evenly across that span by timestamp, reproducing the uniform
     * sampling training used; passing every buffered frame would let the
     * analyser's frame rate, rather than the time base, decide what the model
     * sees.
     *
     * [durationMs] and [fps] describe the raw analysed window, so the card and
     * the diagnostic report what the camera really delivered.
     */
    fun captureSnapshot(): Capture = synchronized(lock) {
        val newline = SystemClock.elapsedRealtime()
        val cutoff = newline - CAPTURE_SPAN_MS
        // Trailing window only: frames older than the span are from before the
        // sign and would dilute it.
        val raw = ring.filter { it.capturedAtMs >= cutoff }.ifEmpty { ring.toList() }
        val first = raw.firstOrNull()?.capturedAtMs ?: 0L
        val last = raw.lastOrNull()?.capturedAtMs ?: 0L
        val duration = if (raw.size >= 2) last - first else 0L
        val fps = if (duration > 0) (raw.size - 1) * 1000f / duration else 0f
        // Report coverage over the frames the model actually receives, so the
        // card's "N frames / M with a hand" describes one window, not two.
        val sampled = evenlySpaced(raw, RING_CAPACITY)
        Capture(sampled, duration, fps, sampled.count { it.hands.isNotEmpty() })
    }

    private fun evenlySpaced(raw: List<Frame>, target: Int): List<Frame> {
        val idx = evenlySpacedIndices(LongArray(raw.size) { raw[it].capturedAtMs }, target)
        return idx.map { raw[it] }
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