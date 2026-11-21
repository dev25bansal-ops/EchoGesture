package org.echogesture.finalapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import androidx.annotation.VisibleForTesting
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
    /** Null only for a test tracker; production construction always supplies one. */
    private val landmarker: HandLandmarker?
) : AutoCloseable {

    private val nextFrameTs = AtomicLong(0L)
    private val closed = AtomicBoolean(false)
    private val ring = ArrayDeque<Frame>()
    private val lock = Any()
    private var captureGeneration = 0
    private val detectorLock = Any()

    /**
     * Real analysed-frame timing.
     *
     * Kept on the tracker rather than derived from capture counts, because a
     * capture's frame count is what survived sampling (capped at 32), not what
     * the camera delivered. Reading the analyser rate off a capture produced the
     * widely-quoted "5-10 fps" figure from a run that was actually capped at 32
     * samples. This is the measurement that answers the question honestly.
     */
    val profiler = AnalysisProfiler()

    /**
     * Explicit record mode (START → sign → STOP). Separate from [ring]: the
     * ring is always rolling, a recording exists only between two presses.
     * Guarded by [lock] like the ring, so a frame can never be offered to a
     * recording while the UI is finalising the previous one.
     */
    private val recorder = SignRecorder()

    /**
     * One-tap capture: decides where a sign starts and ends from the frames
     * themselves, so the signer taps once and the app stops on its own.
     *
     * Kept beside [recorder] rather than replacing it: the manual two-tap path
     * stays as the fallback the brief requires, and both are driven by the same
     * analysed frames so they can never disagree about what the camera saw.
     */
    private val autoSegmenter = AutoSegmenter()

    /** The completed one-tap segment waiting to be classified, if any. */
    private var autoOutcome: AutoSegmenter.Outcome? = null

    /**
     * When the last capture was classified, or [Long.MIN_VALUE] if none yet.
     * A tap too soon after this would re-read the window that still contains
     * the sign just classified, so it is refused rather than answered again.
     */
    private var lastCapturedAtMs = Long.MIN_VALUE

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

    /** Outcome of a capture request. Never carries post-tap frames. */
    sealed interface CaptureRequest {
        data class Ready(val capture: Capture) : CaptureRequest
        data class Insufficient(val reason: Reason, val spanMs: Long = 0L) : CaptureRequest {
            enum class Reason {
                /** No analysed frame at or before the tap. */
                NO_FRAMES,
                /** History exists but covers less than [MIN_USABLE_SPAN_MS]. */
                TOO_SHORT,
                /** Newest frame predates the tap by more than [STALE_GAP_MS]. */
                STALE,
                /** Another sign was read too recently to read a new one. */
                REPEATED,
            }
        }
    }

    companion object {
        private const val TAG = "HandTracker"

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

        /** Minimum buffered history before a tap can be classified at all. */
        const val MIN_USABLE_SPAN_MS = 1200L

        /** A newest frame older than this means the analyser has stalled. */
        const val STALE_GAP_MS = 700L

        /**
         * Minimum real time between two classified captures. Two taps closer
         * than this select overlapping windows from the same sign, so the
         * second would re-report the first sign as if it were a new one.
         */
        const val MIN_REPEAT_GAP_MS = 1200L

        /**
         * The whole capture decision, as a pure function of the buffered frames,
         * the tap instant and the previous capture instant.
         *
         * Tests feed real [Frame] lists with chosen timestamps and get the
         * production answer, rather than re-asserting constants.
         *
         * The contract is strict and has NO waiting path. An earlier version fell
         * back to "keep capturing for another span" when the buffer looked thin,
         * which silently re-introduced the original defect: the user had already
         * finished signing, so the extra window recorded an empty room. If the
         * history is not usable the caller is told so and must ask the user to try
         * again; it is never satisfied with post-tap frames.
         */
        internal fun selectCapture(
            frames: List<Frame>,
            tappedAt: Long,
            previousCaptureAtMs: Long
        ): HandTracker.CaptureRequest {
            if (frames.isEmpty()) {
                return HandTracker.CaptureRequest.Insufficient(HandTracker.CaptureRequest.Insufficient.Reason.NO_FRAMES)
            }
            // Only frames at or before the tap are eligible: a frame that arrived
            // while the tap was being handled belongs to whatever the signer does
            // next, not to the sign. The ring is ordered oldest-first.
            val preTap = frames.filter { it.capturedAtMs <= tappedAt }
            if (preTap.isEmpty()) {
                return HandTracker.CaptureRequest.Insufficient(HandTracker.CaptureRequest.Insufficient.Reason.NO_FRAMES)
            }
            val span = tappedAt - preTap.first().capturedAtMs
            if (span < MIN_USABLE_SPAN_MS) {
                // A cold start, a camera switch, or a paused analyser. Say so
                // rather than classifying a fraction of a sign.
                return HandTracker.CaptureRequest.Insufficient(HandTracker.CaptureRequest.Insufficient.Reason.TOO_SHORT, span)
            }
            val newest = preTap.last().capturedAtMs
            if (tappedAt - newest > STALE_GAP_MS) {
                // The analyser stopped delivering; the newest frame predates the
                // sign, so the window would describe something that is already over.
                return HandTracker.CaptureRequest.Insufficient(HandTracker.CaptureRequest.Insufficient.Reason.STALE)
            }
            // Two taps closer than MIN_REPEAT_GAP_MS select overlapping windows
            // from the same sign, so the second would re-report the first sign as
            // if it were a new one. Refuse rather than answer twice.
            if (previousCaptureAtMs != Long.MIN_VALUE &&
                tappedAt - previousCaptureAtMs < MIN_REPEAT_GAP_MS
            ) {
                return HandTracker.CaptureRequest.Insufficient(
                    HandTracker.CaptureRequest.Insufficient.Reason.REPEATED,
                    tappedAt - previousCaptureAtMs
                )
            }
            return HandTracker.CaptureRequest.Ready(buildWindow(preTap, tappedAt))
        }

    /**
         * Builds the window ending at [tappedAt] from [preTap] (already known to
         * be at or before the tap): the most recent [CAPTURE_SPAN_MS] of it,
         * resampled to [RING_CAPACITY] points. Never falls back to frames
         * outside the span — a thin window is reported as what it is, not
         * padded with older frames.
         */
        internal fun buildWindow(preTap: List<Frame>, tappedAt: Long): Capture {
            val cutoff = tappedAt - CAPTURE_SPAN_MS
            val raw = preTap.filter { it.capturedAtMs >= cutoff }
            val first = raw.firstOrNull()?.capturedAtMs ?: tappedAt
            val last = raw.lastOrNull()?.capturedAtMs ?: tappedAt
            val duration = if (raw.size >= 2) last - first else 0L
            val fps = if (duration > 0) (raw.size - 1) * 1000f / duration else 0f
            val sampled = evenlySpacedFrames(raw, RING_CAPACITY)
            return Capture(sampled, duration, fps, sampled.count { it.hands.isNotEmpty() })
        }

        private fun evenlySpacedFrames(raw: List<Frame>, target: Int): List<Frame> {
            val idx = evenlySpacedIndices(LongArray(raw.size) { raw[it].capturedAtMs }, target)
            return idx.map { raw[it] }
        }

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

        /**
         * A tracker with no landmarker, for tests that drive the segmenter and
         * recorder directly. Production code always goes through [create]; this
         * exists so the capture state machines can be exercised without a
         * camera or a native detector.
         */
        @VisibleForTesting
        fun createForTest(context: Context): HandTracker = HandTracker(landmarker = null)

        fun create(context: Context, modelAssetPath: String): HandTracker {
            return HandTracker(createLandmarker(context, modelAssetPath, RunningMode.VIDEO))
        }

        internal fun createLandmarker(context: Context, modelAssetPath: String, mode: RunningMode): HandLandmarker {
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder().setModelAssetPath(modelAssetPath).build()
                )
                .setRunningMode(mode)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.3f)
                .setMinHandPresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .build()
            return HandLandmarker.createFromOptions(context, options)
        }

        internal fun toHands(result: HandLandmarkerResult, verticalScale: Float = 1f): List<Hand> =
            result.landmarks().mapIndexed { index, points ->
                val handedness = result.handedness().getOrNull(index)?.firstOrNull()?.categoryName()
                val slot = if (handedness.equals("Left", ignoreCase = true)) 0 else 1
                val xyz = FloatArray(21 * 3)
                for (joint in 0 until 21) {
                    xyz[joint * 3] = points[joint].x()
                    xyz[joint * 3 + 1] = points[joint].y() * verticalScale
                    xyz[joint * 3 + 2] = points[joint].z()
                }
                Hand(slot, xyz)
            }
    }

    /**
     * Analyses one camera frame. Returns true if the frame was appended, false
     * otherwise. [ImageProxy.close] is guaranteed exactly once on every path,
     * including a closed tracker. Safe to call from the ImageAnalysis executor.
     */
    fun analyse(
        imageProxy: ImageProxy,
        profileFrame: Boolean = true,
        isCurrentStream: () -> Boolean = { true }
    ): Boolean {
        val generation = synchronized(lock) { captureGeneration }
        if (closed.get() || !isCurrentStream()) {
            imageProxy.close()
            return false
        }
        var mpImage: MPImage? = null
        // Timestamp when the frame ARRIVED, not when detection finished.
        // detectForVideo is synchronous and can take tens of milliseconds; a
        // frame that arrived before the user tapped must not be credited to
        // the capture window just because its analysis finished after it.
        val frameAtMs = SystemClock.elapsedRealtime()
        var convertMs = 0L
        var detectMs = 0L
        return try {
            // MediaPipe reports landmarks in the input bitmap's coordinates.
            // Rotate the pixels so the gesture features share the preview's
            // upright axes, even when CameraX supplies a sideways sensor buffer.
            val convertStart = SystemClock.elapsedRealtime()
            val raw = imageProxy.toBitmap()
            val rotation = imageProxy.imageInfo.rotationDegrees
            val bitmap = if (rotation == 0) raw else try {
                Bitmap.createBitmap(
                    raw, 0, 0, raw.width, raw.height,
                    Matrix().apply { postRotate(rotation.toFloat()) }, false
                )
            } finally {
                raw.recycle()
            }
            convertMs = SystemClock.elapsedRealtime() - convertStart
            mpImage = try {
                BitmapImageBuilder(bitmap).build()
            } catch (error: RuntimeException) {
                bitmap.recycle()
                throw error
            }
            // detectForVideo() is synchronous: by the time it returns MediaPipe
            // has copied the pixels into its packet, so closing mpImage (which
            // recycles `bitmap`) afterwards is safe.
            val detectStart = SystemClock.elapsedRealtime()
            val detector = landmarker
                ?: throw IllegalStateException("no landmarker: test tracker cannot analyse")
            val result = synchronized(detectorLock) {
                if (closed.get()) return false
                detector.detectForVideo(
                    mpImage,
                    nextFrameTs.updateAndGet { previous -> maxOf(previous + 1, frameAtMs) }
                )
            }
            detectMs = SystemClock.elapsedRealtime() - detectStart
            // MediaPipe makes x and depth relative to image width and y relative
            // to height. Express y in the model's 4:3 reference geometry, keeping
            // all three axes in proportion after portrait or widescreen capture.
            val verticalScale = (bitmap.height.toFloat() / bitmap.width) * (4f / 3f)
            val frame = Frame(toHands(result, verticalScale), frameAtMs)
            // The ring is a rolling buffer of the recent past. It is never
            // gated on a capture state: the sign happens BEFORE the tap, so the
            // frames that describe it are already behind us when the user
            // presses the button. Discarding pre-tap frames here (as an earlier
            // version did) left the window holding only the seconds AFTER the
            // sign — usually an empty room.
            synchronized(lock) {
                if (generation != captureGeneration || !isCurrentStream()) return false
                if (profileFrame) {
                    profiler.record(frameAtMs, convertMs, detectMs, imageProxy.imageInfo.timestamp)
                }
                ring.addLast(frame)
                while (ring.size > MAX_CAPTURE_FRAMES) ring.removeFirst()
                // Offer the SAME frame to an in-progress recording. One
                // analysed frame, one place it arrives, so quick scan and
                // record mode can never see different sequences.
                recorder.offer(frame)

                // One-tap path. offer() returns a non-null Outcome exactly once,
                // on the frame where the sign is judged complete; that is handed
                // to the UI thread, which classifies it immediately. A sign is
                // never classified twice because the outcome is consumed here.
                val done = autoSegmenter.offer(frame)
                if (done != null) autoOutcome = done
                consecutiveFailures.set(0)
            }
            true
        } catch (t: Throwable) {
            // A single bad frame must not stop the preview, but a run of them
            // means the detector is broken. Count consecutive failures so the
            // UI can surface a real error instead of a silently empty preview.
            // Only the exception type is logged — never frame or image data.
            synchronized(lock) {
                if (generation == captureGeneration && isCurrentStream()) {
                    consecutiveFailures.incrementAndGet()
                    Log.w(TAG, "Hand analysis failed (${t.javaClass.simpleName})")
                }
            }
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

    fun latestFrameHasHands(): Boolean = synchronized(lock) {
        ring.lastOrNull()?.hands?.isNotEmpty() == true
    }

    /**
     * Drops buffered frames. Also forgets the repeat guard: a camera switch or
     * lifecycle reset is a new context, not a rapid re-tap.
     *
     * A camera switch mid-recording therefore ends it: the frames before and
     * after a rebind come from different sensors and must never be spliced
     * into one sequence.
     */
    fun clear() = synchronized(lock) {
        captureGeneration++
        consecutiveFailures.set(0)
        ring.clear()
        lastCapturedAtMs = Long.MIN_VALUE
        recorder.cancel()
        autoOutcome = null
        autoSegmenter.cancel()
    }

    // ---- one-tap automatic capture -----------------------------------------

    /**
     * Arms the one-tap segmenter at [nowMs]. Any half-finished attempt and any
     * frame from the previous sign are discarded, so a new tap can never be
     * classified using the last sign's frames.
     */
    fun armAuto(nowMs: Long): Unit = synchronized(lock) {
        autoOutcome = null
        profiler.reset()
        val last = ring.lastOrNull()
        val handAlreadyPresent = last != null && last.hands.isNotEmpty() &&
            nowMs - last.capturedAtMs in 0L..500L
        autoSegmenter.arm(nowMs, handAlreadyPresent)
    }

    /** Disarms without classifying. Used when the user leaves the tab. */
    fun disarmAuto(): Unit = synchronized(lock) {
        autoOutcome = null
        autoSegmenter.cancel()
    }

    /** Ends the current attempt now, keeping everything captured so far. */
    fun finishAutoManually(nowMs: Long): AutoSegmenter.Outcome? = synchronized(lock) {
        val completed = autoOutcome
        autoOutcome = null
        completed ?: if (autoSegmenter.isActive) autoSegmenter.finishManually(nowMs) else null
    }

    fun expireAuto(nowMs: Long): AutoSegmenter.Outcome? = synchronized(lock) {
        val completed = autoOutcome
        autoOutcome = null
        completed ?: if (autoSegmenter.isActive) autoSegmenter.expire(nowMs) else null
    }

    /**
     * Hands back the completed segment exactly once.
     *
     * Consumed rather than read, so a segment the segmenter judged complete can
     * never be classified twice — the UI calls this from the analyser callback
     * and acts only on a non-null return.
     */
    fun takeAutoOutcome(): AutoSegmenter.Outcome? = synchronized(lock) {
        val o = autoOutcome
        autoOutcome = null
        o
    }

    fun autoState(): AutoSegmenter.State = synchronized(lock) { autoSegmenter.state }

    /**
     * Offers an already-built frame to the same capture machinery [analyse]
     * feeds, without an ImageProxy or a detector.
     *
     * Test-only: production frames always arrive through [analyse], so the two
     * can never drift in behaviour because this method shares that code path's
     * lock and data structures.
     */
    @VisibleForTesting
    fun offerForTest(frame: Frame) = synchronized(lock) {
        ring.addLast(frame)
        while (ring.size > MAX_CAPTURE_FRAMES) ring.removeFirst()
        recorder.offer(frame)
        autoSegmenter.offer(frame)?.let { autoOutcome = it }
    }

    fun autoFrameCount(): Int = synchronized(lock) { autoSegmenter.frameCount() }

    // ---- explicit recording mode -------------------------------------------

    /** Starts a fresh recording at [nowMs]. Discards any in-progress one. */
    fun startRecording(nowMs: Long): Unit = synchronized(lock) {
        profiler.reset()
        recorder.start(nowMs)
    }

    fun isRecording(): Boolean = synchronized(lock) { recorder.isRecording }

    fun recordingStartedAt(): Long? = synchronized(lock) { recorder.startedAt() }

    fun recordingFrameCount(): Int = synchronized(lock) { recorder.frameCount() }

    fun shouldAutoStopRecording(nowMs: Long): Boolean =
        synchronized(lock) { recorder.shouldAutoStop(nowMs) }

    fun cancelRecording(): Unit = synchronized(lock) { recorder.cancel() }

    /** Ends the recording at [nowMs] and reports whether it is usable. */
    fun stopRecording(nowMs: Long): SignRecorder.Result =
        synchronized(lock) { recorder.stop(nowMs) }

    /**
     * Classifies the sign that just happened.
     *
     * The signer performs, THEN taps, so the frames describing the sign are
     * already in the rolling ring. The window is the [CAPTURE_SPAN_MS] of real
     * time ending at the tap — never anything after it.
     *
     * Returns the outcome synchronously so the UI can react immediately; the
     * decision itself lives in [selectCapture], which is pure.
     */
    fun beginCapture(): HandTracker.CaptureRequest {
        val tappedAt = SystemClock.elapsedRealtime()
        synchronized(lock) {
            val outcome = selectCapture(ring.toList(), tappedAt, lastCapturedAtMs)
            if (outcome is CaptureRequest.Ready) lastCapturedAtMs = tappedAt
            return outcome
        }
    }

    private fun evenlySpaced(raw: List<Frame>, target: Int): List<Frame> {
        val idx = evenlySpacedIndices(LongArray(raw.size) { raw[it].capturedAtMs }, target)
        return idx.map { raw[it] }
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            synchronized(detectorLock) { landmarker?.close() }
            clear()
        }
    }

}
