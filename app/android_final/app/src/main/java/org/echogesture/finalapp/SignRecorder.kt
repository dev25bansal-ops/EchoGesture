package org.echogesture.finalapp

/**
 * The explicit recording mode: START → perform one sign → STOP.
 *
 * This is the mode that matches how the training clips were made. A training
 * clip *is* one sign, start to finish, and `read_frames()` samples 32 frames
 * evenly across that whole clip. The quick-scan mode has to guess where the
 * sign was inside a fixed trailing window; here the signer says where it was.
 *
 * Deliberately separate from [HandTracker]'s rolling ring. The ring answers
 * "what happened in the last 3.7 s" and is always running; a recording answers
 * "what happened between these two instants" and exists only while recording.
 * Keeping them apart means the ring cannot leak pre-start frames into a
 * recording, and a recording cannot disturb quick scan.
 *
 * Pure Kotlin: no Android types, no MediaPipe, no clock. The caller supplies
 * each frame with the instant it ARRIVED. That makes the whole start/stop
 * contract testable with deterministic timestamps, which is the only way to
 * prove "no pre-start frame, no post-stop frame" rather than assert it.
 */
class SignRecorder(
    /** Hard cap on stored frames. ~3.5 min at 30 fps; bounds memory. */
    private val maxFrames: Int = MAX_FRAMES_DEFAULT
) {

    /** A recorded interval. [frames] are in arrival order, oldest first. */
    data class Recording(
        val frames: List<HandTracker.Frame>,
        val startedAtMs: Long,
        val stoppedAtMs: Long
    ) {
        val frameCount: Int get() = frames.size
        val durationMs: Long get() = stoppedAtMs - startedAtMs
    }

    /** Why a recording could not be used. */
    enum class Rejection {
        /** Stop arrived before Start. */
        NOT_STARTED,
        /** The interval holds no analysed frame. */
        NO_FRAMES,
        /** The interval is shorter than [MIN_DURATION_MS]. */
        TOO_SHORT,
        /** The interval ran past [MAX_DURATION_MS]; recording auto-stopped. */
        TOO_LONG,
        /** The camera delivered nothing for longer than [STALL_MS]. */
        CAMERA_STALLED,
        /** Too few analysed frames carry a hand. */
        LOW_COVERAGE,
    }

    sealed interface Result {
        data class Usable(val recording: Recording) : Result
        data class Unusable(
            val reason: Rejection,
            val recording: Recording? = null
        ) : Result
    }

    private val frames = ArrayList<HandTracker.Frame>(256)
    private var startedAtMs = Long.MIN_VALUE
    private var truncated = false

    /** True between [start] and [stop]. */
    val isRecording: Boolean get() = startedAtMs != Long.MIN_VALUE

    /** When the current recording began, or null. */
    fun startedAt(): Long? = if (isRecording) startedAtMs else null

    /** Frames captured so far in the current recording. */
    fun frameCount(): Int = frames.size

    /**
     * Begins a new recording at [nowMs]. Any previous recording is discarded —
     * audio-style "start while started" would silently splice two attempts.
     */
    fun start(nowMs: Long) {
        frames.clear()
        startedAtMs = nowMs
        truncated = false
    }

    /**
     * Called by the analyser for every frame. Returns true if the frame was
     * stored. Frames that arrive before Start or after the auto-stop are
     * dropped, so a recording never contains a frame from outside its interval.
     */
    fun offer(frame: HandTracker.Frame): Boolean {
        if (!isRecording) return false
        // Auto-stop on the cap rather than dropping the newest frames: a sign
        // that runs long is truncated at its END, never at its beginning, so
        // the onset the model needs is always present.
        if (frames.size >= maxFrames) {
            truncated = true
            return false
        }
        frames.add(frame)
        return true
    }

    /** True once [MAX_DURATION_MS] has elapsed since Start. */
    fun shouldAutoStop(nowMs: Long): Boolean =
        isRecording && nowMs - startedAtMs >= MAX_DURATION_MS

    /** True if the buffer hit its frame cap during this recording. */
    fun wasTruncated(): Boolean = truncated

    /** Discards the in-progress recording. */
    fun cancel() {
        frames.clear()
        startedAtMs = Long.MIN_VALUE
        truncated = false
    }

    /**
     * Ends the recording at [nowMs] and decides whether it is usable.
     *
     * Only frames whose arrival instant is inside `[startedAtMs, nowMs]` are
     * kept. That is belt-and-braces given [offer] already refuses out-of-window
     * frames, but it is the property the whole mode rests on, so it is enforced
     * where the recording is finalised as well.
     */
    fun stop(nowMs: Long): Result {
        if (!isRecording) return Result.Unusable(Rejection.NOT_STARTED)
        val began = startedAtMs
        startedAtMs = Long.MIN_VALUE
        val inside = frames.filter { it.capturedAtMs in began..nowMs }
        val recording = Recording(inside, began, nowMs)
        frames.clear()

        val tooLong = truncated || nowMs - began > MAX_DURATION_MS
        return when {
            inside.isEmpty() -> Result.Unusable(Rejection.NO_FRAMES, recording)
            nowMs - began < MIN_DURATION_MS -> Result.Unusable(Rejection.TOO_SHORT, recording)
            tooLong -> Result.Unusable(Rejection.TOO_LONG, recording)
            else -> {
                val stalled = longestGapMs(inside)
                val coverage = inside.count { it.hands.isNotEmpty() }.toFloat() / inside.size
                when {
                    // A long hole means the camera stopped mid-sign; the
                    // sequence has a break in it rather than being merely thin.
                    stalled > STALL_MS -> Result.Unusable(Rejection.CAMERA_STALLED, recording)
                    coverage < MIN_HAND_COVERAGE ->
                        Result.Unusable(Rejection.LOW_COVERAGE, recording)
                    else -> Result.Usable(recording)
                }
            }
        }
    }

    companion object {
        /**
         * Bounds on a recording. Training clips are a single sign: ASL Citizen
         * clips run roughly 1–6 s, and the extractor's own docstring records a
         * typical 110-frame clip as about 3.7 s. The range below brackets that
         * generously rather than assuming every sign needs exactly 3.7 s.
         *
         * These are capture bounds, not model bounds: whatever interval is
         * recorded is resampled to the model's 24 steps afterwards, so a 1.5 s
         * sign and a 6 s sign both reach the model as 24 steps — exactly as a
         * short and a long training clip both became 32 samples.
         */
        const val MIN_DURATION_MS = 400L
        const val MAX_DURATION_MS = 12_000L

        /**
         * Largest gap between consecutive analysed frames that still counts as
         * continuous capture. Above this the camera dropped out mid-sign and
         * the sequence has a hole.
         */
        const val STALL_MS = 700L

        /**
         * Fraction of recorded frames that must carry a hand.
         *
         * Lower than quick scan's 0.15 on purpose. An explicit recording is
         * bracketed by the signer's own button presses, so the opening and
         * closing moments are often "hand entering frame"; and some signs are
         * made with one hand while the other is out of shot. A static-looking
         * but real sign must not be rejected for low *motion* — coverage is
         * about whether the hand was visible, not whether it moved.
         */
        const val MIN_HAND_COVERAGE = 0.10f

        /** ~3.5 minutes at 30 fps. Bounds memory at roughly 3 MB of floats. */
        const val MAX_FRAMES_DEFAULT = 6400

        /** Longest gap between consecutive frames, in ms; 0 if fewer than 2. */
        fun longestGapMs(frames: List<HandTracker.Frame>): Long {
            var worst = 0L
            for (i in 1 until frames.size) {
                val gap = frames[i].capturedAtMs - frames[i - 1].capturedAtMs
                if (gap > worst) worst = gap
            }
            return worst
        }
    }
}