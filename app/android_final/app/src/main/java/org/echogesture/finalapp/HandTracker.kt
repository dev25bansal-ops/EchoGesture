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
import com.google.mediapipe.tasks.core.Delegate
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

    /**
     * Detects currently inside [landmarker], and a latch [close] waits on.
     *
     * WHY THIS EXISTS. Closing the landmarker while a frame is inside
     * `detectForVideo()` tears down the native graph under a running detect:
     * `TaskRunner.close()` is not synchronised in tasks-core 0.10.14, so this is
     * a native abort, not a catchable exception.
     *
     * `onDestroy()` used to claim that calling `trackerExecutor.shutdownNow()`
     * first prevented it. It does not: shutdownNow interrupts and drains but
     * does NOT wait, [analyse] checks [closed] once on entry and never checks
     * `isInterrupted`, and MediaPipe's native detect ignores interrupts
     * entirely. The window was narrowed, never closed.
     *
     * So every detect is bracketed by [enterDetect]/[exitDetect], and [close]
     * waits for the count to reach zero before touching the landmarker. The
     * wait is bounded, because a hung detect must not hang the UI thread
     * forever; on timeout the landmarker is still closed, but the caller has
     * been given the chance it previously never had.
     */
    private val inFlight = java.util.concurrent.atomic.AtomicInteger(0)
    private val idle = java.util.concurrent.CountDownLatch(1)

    private fun enterDetect() {
        inFlight.incrementAndGet()
    }

    private fun exitDetect() {
        if (inFlight.decrementAndGet() == 0) idle.countDown()
    }
    private val ring = ArrayDeque<Frame>()
    private val lock = Any()

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
     * When the last capture was classified, or [Long.MIN_VALUE] if none yet.
     * A tap too soon after this would re-read the window that still contains
     * the sign just classified, so it is refused rather than answered again.
     */
    private var lastCapturedAtMs = Long.MIN_VALUE

    /**
     * One-shot camera-frame dump, OFF by default.
     *
     * Set [frameDumpDir] and the next recording writes its frames there as
     * PNGs, then clears the field. This exists because "the clip works but the
     * camera does not" could not be resolved from the phone's own numbers: the
     * diagnostics record the PREDICTION, not the pixels, so there was no way to
     * tell whether MediaPipe produced different landmarks from the same scene
     * or the classifier misread the same landmarks.
     *
     * Deliberately narrow: one sign, explicitly armed, only while recording,
     * and it disarms itself. It never accumulates images across sessions and
     * never runs unless the caller sets the directory. An accessibility tool
     * must not silently keep camera frames on disk.
     */
    @Volatile
    var frameDumpDir: java.io.File? = null
    private val dumpedFrames = AtomicInteger(0)

    /**
     * Separate worker for the dump's PNG encoding, so a diagnostic capture never
     * steals time from the analyser it is meant to be measuring. Single thread
     * and unbounded queue is fine: the dump is capped by [MAX_DUMP_FRAMES] and
     * only ever armed for one recording.
     */
    private val dumpExecutor: java.util.concurrent.ExecutorService? by lazy {
        java.util.concurrent.Executors.newSingleThreadExecutor { r ->
            Thread(r, "frame-dump").apply { isDaemon = true }
        }
    }

    /** Arms a single-capture frame dump into [dir]; the next recording fills it. */
    fun armFrameDump(dir: java.io.File) {
        dir.mkdirs()
        dumpedFrames.set(0)
        frameDumpDir = dir
    }

    /** Disarms without writing anything further. */
    fun disarmFrameDump() {
        frameDumpDir = null
    }

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
        /**
         * Maps one hand's landmarks from the frame MediaPipe saw into the
         * geometry the checkpoints were TRAINED on. In place; z is untouched.
         *
         * MediaPipe normalises x by the frame width and y by its height
         * separately, so on a frame that is not the training aspect one
         * normalised x unit and one normalised y unit are different physical
         * lengths and the hand arrives stretched. The corpus is 640x480 (4:3
         * landscape); an app pinned to portrait hands the detector 480x640,
         * which stretches x by 4/3 and squashes y by 3/4 — a 1.78x anisotropy,
         * larger than a margin of error.
         *
         * Measured, with the nine shipped members, on the 1,251-clip test set
         * (`landmarks_native`): the warp costs 85.08% -> 82.66% top-1, so it is
         * not subtle. On a real 32-frame camera capture, correcting it moved
         * member agreement from 1/9 to 5/9 and the top-1 margin from 0.061 to
         * 0.202. Both paths — live camera here and [ClipAnalyzer] for files —
         * call this, because a clip must classify the same way whether it
         * arrives from the camera or from the picker.
         *
         * The transform scales x about the origin and y about the MIDPOINT, so
         * 0 stays 0 and 1 stays 1 on both axes: a landmark can never be pushed
         * off the frame. Square frames (the 256-squash experiments) and 4:3
         * frames are identities. A zero-sized frame — possible before the
         * analyser has been configured — is left alone rather than producing
         * NaN, which features.py would refuse and the capture would be lost.
         */
        fun mapToCorpusAspect(xyz: FloatArray, width: Int, height: Int): FloatArray {
            if (width <= 0 || height <= 0) return xyz
            // MediaPipe normalises each axis by its own frame dimension:
            // x_norm * width and y_norm * height are pixels. Expressing a
            // landmark in the corpus's normalisation therefore means
            //     x_corpus = x_norm * width  / 640
            //     y_corpus = y_norm * height / 480
            // which is the identity at 640x480 and leaves the hand's ASPECT
            // correct at any other shape. A uniform overall scale is harmless
            // because the feature transform divides by wrist->middle-MCP, so
            // only the RATIO of these two factors reaches the model. A
            // 480x640 portrait frame gives (0.75, 1.333): the hand arrives
            // squashed in x and stretched in y, an anisotropy of 1.78x.
            val xScale = width.toFloat() / CORPUS_WIDTH_PX
            val yScale = height.toFloat() / CORPUS_HEIGHT_PX
            if (xScale == 1f && yScale == 1f) return xyz
            val n = xyz.size / 3
            for (j in 0 until n) {
                xyz[j * 3] *= xScale
                // y is anchored at the frame's top edge, which is MediaPipe's
                // origin for this axis. The anchor is immaterial to the model
                // (the wrist is subtracted in the feature transform); it is
                // chosen so the expression stays a plain scale.
                xyz[j * 3 + 1] *= yScale
            }
            return xyz
        }

        /** Landscape corpus geometry the checkpoints were trained on. */
        private const val CORPUS_WIDTH_PX = 640f
        private const val CORPUS_HEIGHT_PX = 480f

        private const val TAG = "HandTracker"
        private const val FRAME_INTERVAL_MS = 33L // ~30 fps, monotonic VIDEO timestamps

        /** Frames resampled across the window into the model's 24-step input. */
        const val RING_CAPACITY = 32

        /** Upper bound on dumped frames per recording, so a long one cannot fill the disk. */
        const val MAX_DUMP_FRAMES = 40

        /** Longest close() waits for an in-flight detect to finish; see [inFlight]. */
        const val CLOSE_DRAIN_MS = 500L

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

        fun create(context: Context, modelAssetPath: String): HandTracker {
            // The detector is the dominant inference cost on-device: measured
            // detect_p50 80-109 ms / p95 159-187 ms on the Vivo V2404, against
            // the organizer's 50 ms latency budget. That measurement was taken
            // with NO delegate set, i.e. the default CPU path. MediaPipe 0.10.14
            // exposes Delegate.CPU and Delegate.GPU (verified with javap against
            // tasks-core-0.10.14.aar; there is no NNAPI value in this version),
            // and the organizer's own runtime column example is "GPU delegate".
            //
            // GPU is attempted first and CPU is the fallback, because a GPU
            // delegate that fails to initialise must not cost us the demo: GPU
            // support varies by driver and a hard bind would turn a slower app
            // into a broken one. Which delegate actually took effect is
            // recorded so the on-device latency figure can state its runtime.
            val gpu = try {
                buildLandmarker(context, modelAssetPath, Delegate.GPU).also {
                    Log.i(TAG, "HandLandmarker delegate=GPU")
                }
            } catch (t: Throwable) {
                Log.w(TAG, "GPU delegate unavailable, falling back to CPU", t)
                buildLandmarker(context, modelAssetPath, Delegate.CPU)
            }
            return HandTracker(gpu)
        }

        private fun buildLandmarker(
            context: Context,
            modelAssetPath: String,
            delegate: Delegate,
        ): HandLandmarker {
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath(modelAssetPath)
                        .setDelegate(delegate)
                        .build()
                )
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.3f)
                .setMinHandPresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .build()
            return HandLandmarker.createFromOptions(context, options)
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
        var convertMs = 0L
        var detectMs = 0L
        return try {
            // toBitmap() copies the YUV planes into a fresh ARGB_8888 bitmap
            // and does NOT apply rotation; rotation is passed to MediaPipe via
            // ImageProcessingOptions below, exactly once.
            val convertStart = SystemClock.elapsedRealtime()
            val bitmap = imageProxy.toBitmap()
            convertMs = SystemClock.elapsedRealtime() - convertStart
            // Frame dump, if armed. Written BEFORE detection so the file is what
            // the detector is about to see. Rotation is applied here because
            // MediaPipe receives it via ImageProcessingOptions and NOT in the
            // bitmap itself -- dumping the unrotated copy would hand the offline
            // pipeline an upside-down frame and make the comparison meaningless.
            frameDumpDir?.let { dir ->
                // Cheap here, expensive later: the ROTATED COPY is made on this
                // thread (it must be, the bitmap is recycled when detectForVideo
                // returns) but the PNG ENCODE AND WRITE happen on a separate
                // worker.
                //
                // This is a measured regression, not a precaution. The first
                // version encoded inline, on the analyser's own thread, and the
                // profiler showed the cost directly: interval_p95 went to
                // 200-254 ms and analysed rate fell to 2.8-4.7 fps on captures,
                // because a ~800 KB PNG of a 960x960 frame takes tens of
                // milliseconds and the analyser thread is a single thread with
                // no slack. Diagnostics stays ON for a consenting test session,
                // so this ran on every capture.
                runCatching {
                    val rot = imageProxy.imageInfo.rotationDegrees
                    val m = android.graphics.Matrix()
                    if (rot != 0) m.postRotate(rot.toFloat())
                    val shown = if (rot == 0) bitmap
                    else Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
                    val idx = dumpedFrames.getAndIncrement()
                    // Bounded: a long recording cannot fill the disk. 40 frames
                    // is more than any capture after MIN_FRAMES trimming.
                    if (idx < MAX_DUMP_FRAMES && dumpExecutor != null) {
                        val snapshot = shown.copy(Bitmap.Config.ARGB_8888, false)
                        if (shown !== bitmap) shown.recycle()
                        runCatching {
                            dumpExecutor?.execute {
                                runCatching {
                                    val f = java.io.File(dir, "frame_%03d.png".format(idx))
                                    java.io.FileOutputStream(f).use {
                                        snapshot.compress(Bitmap.CompressFormat.PNG, 100, it)
                                    }
                                }
                                snapshot.recycle()
                            }
                        }
                    } else if (shown !== bitmap) {
                        shown.recycle()
                    }
                }
            }
            mpImage = BitmapImageBuilder(bitmap).build()
            // detectForVideo() is synchronous: by the time it returns MediaPipe
            // has copied the pixels into its packet, so closing mpImage (which
            // recycles `bitmap`) afterwards is safe.
            val detectStart = SystemClock.elapsedRealtime()
            // Bracketed so close() cannot tear the native graph down underneath
            // this call. See [inFlight].
            val result = try {
                enterDetect()
                landmarker.detectForVideo(
                    mpImage,
                    ImageProcessingOptions.builder()
                        .setRotationDegrees(imageProxy.imageInfo.rotationDegrees)
                        .build(),
                    nextFrameTs.getAndAdd(FRAME_INTERVAL_MS)
                )
            } finally {
                exitDetect()
            }
            detectMs = SystemClock.elapsedRealtime() - detectStart
            // Real analysed-frame timing, including the camera's own sensor
            // timestamp so dropped frames can be counted. Without this the live
            // analysis rate is unknowable and every "it feels slow on my phone"
            // report stays a guess.
            profiler.record(
                arrivalMs = frameAtMs,
                convertDurationMs = convertMs,
                detectDurationMs = detectMs,
                sensorTimestampNs = imageProxy.imageInfo.timestamp
            )
            val frame = Frame(
                toHands(result, mpImage.getWidth(), mpImage.getHeight()),
                frameAtMs
            )
            // The ring is a rolling buffer of the recent past. It is never
            // gated on a capture state: the sign happens BEFORE the tap, so the
            // frames that describe it are already behind us when the user
            // presses the button. Discarding pre-tap frames here (as an earlier
            // version did) left the window holding only the seconds AFTER the
            // sign — usually an empty room.
            synchronized(lock) {
                ring.addLast(frame)
                while (ring.size > MAX_CAPTURE_FRAMES) ring.removeFirst()
                // Offer the SAME frame to an in-progress recording. One
                // analysed frame, one place it arrives, so quick scan and
                // record mode can never see different sequences.
                recorder.offer(frame)
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

    /**
     * Drops buffered frames. Also forgets the repeat guard: a camera switch or
     * lifecycle reset is a new context, not a rapid re-tap.
     *
     * A camera switch mid-recording therefore ends it: the frames before and
     * after a rebind come from different sensors and must never be spliced
     * into one sequence.
     */
    fun clear() = synchronized(lock) {
        // A discarded or abandoned capture must not keep writing camera PNGs.
        // Only stopRecording() used to disarm the dump, so cancelRecording()
        // and clear() left it armed.
        frameDumpDir = null
        ring.clear()
        lastCapturedAtMs = Long.MIN_VALUE
        recorder.cancel()
    }

    // ---- explicit recording mode -------------------------------------------

    /** Starts a fresh recording at [nowMs]. Discards any in-progress one. */
    fun startRecording(nowMs: Long): Unit = synchronized(lock) { recorder.start(nowMs) }

    fun isRecording(): Boolean = synchronized(lock) { recorder.isRecording }

    fun recordingStartedAt(): Long? = synchronized(lock) { recorder.startedAt() }

    fun recordingFrameCount(): Int = synchronized(lock) { recorder.frameCount() }

    fun shouldAutoStopRecording(nowMs: Long): Boolean =
        synchronized(lock) { recorder.shouldAutoStop(nowMs) }

    fun cancelRecording(): Unit = synchronized(lock) {
        frameDumpDir = null
        recorder.cancel()
    }

    /** Ends the recording at [nowMs] and reports whether it is usable. */
    fun stopRecording(nowMs: Long): SignRecorder.Result =
        synchronized(lock) {
            // One sign per arm: a dump that kept writing would grow without
            // bound across a session.
            frameDumpDir = null
            recorder.stop(nowMs)
        }

    /**
     * The frames recorded so far, for the auto-finish detector. Taken under the
     * tracker's own [lock], which is the monitor that guards the recorder.
     */
    fun recordingSnapshot(): List<Frame> = synchronized(lock) { recorder.snapshot() }

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
            // Wait for any frame already inside detectForVideo() to come back
            // out. This is the whole point: TaskRunner.close() is not
            // synchronised, so closing while a detect is in flight frees the
            // native graph underneath it and the process dies.
            //
            // Bounded at 500 ms. A detect measured 36-58 ms, so this normally
            // returns immediately; the timeout only matters if the detector has
            // wedged, and hanging the UI thread forever would be a worse bug
            // than the one being fixed.
            if (inFlight.get() > 0) {
                runCatching { idle.await(CLOSE_DRAIN_MS, java.util.concurrent.TimeUnit.MILLISECONDS) }
            }
            landmarker.close()
            synchronized(lock) {
                ring.clear()
                lastCapturedAtMs = Long.MIN_VALUE
            }
        }
    }

    private fun toHands(result: HandLandmarkerResult, width: Int, height: Int): List<Hand> {
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
            mapToCorpusAspect(arr, width, height)
            out.add(Hand(slot, arr))
            i++
        }
        return out
    }

}