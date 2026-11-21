package org.echogesture.finalapp

/**
 * Decides where a sign starts and stops, from the frames themselves.
 *
 * Why this exists: the shipped flow asks the signer to bracket a sign with two
 * taps. A real phone recording shows what that costs — the signer taps Start,
 * waits, signs, then hunts for Finish. Everything between the taps is recorded,
 * so idle before the sign and idle after it are spread across the 32 samples the
 * model gets, and the model sees a diluted version of the motion it was trained
 * on. Worse, the signer has to know when their own sign ended.
 *
 * This is a state machine over analysed frames. It answers three questions:
 * has a sign started, has it finished, and which frames belong to it. The signer
 * taps once; this does the rest.
 *
 * Everything here is pure: no Android types, no clock, no MediaPipe. The caller
 * supplies each frame with the instant it ARRIVED, so the whole contract is
 * testable with synthetic sequences and exact timestamps.
 *
 * ## Why the thresholds look the way they do
 *
 * **Time, not frames.** The phone's analyser delivers 5-10 fps, not 30. A rule
 * expressed in frames ("stop after 10 quiet frames") therefore means a
 * different duration on every device and on every thermal state. Every threshold
 * below is milliseconds.
 *
 * **Motion is relative, not absolute.** Landmark coordinates are normalised by
 * hand size, so "0.02" means a different physical movement for a hand near the
 * lens and one far away. The onset test compares motion to the signer's own
 * recent noise floor instead of a fixed constant.
 *
 * **Stationary signs are real.** Many isolated signs hold a handshape with
 * almost no movement. A segmenter keyed only on motion would never start on
 * them. So a hand appearing and holding a stable pose for [POSE_HOLD_MS] also
 * arms a sign, at a lower motion bar.
 *
 * **Pauses inside signs are real.** Two-part signs have a still moment in the
 * middle. The end test requires motion to stay below the floor for
 * [QUIET_HOLD_MS] *continuously*; any motion above the floor restarts that
 * timer.
 */
class AutoSegmenter(
    private val config: Config = Config()
) {

    companion object {
        /**
         * Hard bound on buffered frames.
         *
         * 600 frames is ~20 s at 30 fps, comfortably above [Config.maxSignMs],
         * so it never truncates a normal attempt. It exists so a fast device or
         * a clock anomaly cannot turn one attempt into unbounded memory growth.
         */
        const val MAX_BUFFERED_FRAMES = 600
    }

    data class Config(
        /**
         * Hand must be present for this long before a sign can start. Guards
         * against one noisy detection arming a capture, and gives the signer
         * time to settle into frame.
         */
        val acquireMs: Long = 300L,
        /**
         * A sign also counts as started when the hand holds a stable pose this
         * long, even with little movement. Static signs are common and a
         * motion-only onset would never fire on them.
         */
        val poseHoldMs: Long = 450L,
        /**
         * How much motion above the noise floor confirms an intentional sign,
         * measured over the trailing window. Deliberately short: waiting for a
         * big movement loses the onset, which is the part carrying the initial
         * handshape.
         */
        val onsetConfirmMs: Long = 200L,
        /**
         * Quiet time that ends a sign. Long enough to bridge a pause inside a
         * two-part sign, short enough that the signer is not left waiting.
         */
        val quietHoldMs: Long = 800L,
        /** Frames kept before the detected onset so the handshape is not lost. */
        val preRollMs: Long = 550L,
        /** Frames kept after the quiet period, matching a manual stop. */
        val postRollMs: Long = 200L,
        /** Shorter than this and the segment is a fragment, not a sign. */
        val minSignMs: Long = 500L,
        /** No sign can start until this long after arming. */
        val armingMs: Long = 250L,
        /** Hard cap; the segmenter stops a sign that never ends. */
        val maxSignMs: Long = 10_000L,
        /** Hand absent this long while signing ends the sign as interrupted. */
        val handLossGraceMs: Long = 900L,
        /** Motion this far above the floor counts as signing activity. */
        val motionFloorMultiplier: Float = 1.8f,
        /** Assumed motion floor before enough frames exist to measure one. */
        val initialFloor: Float = 0.004f,
        /** Lower bound on the measured floor, so a perfectly still hand cannot set it to zero. */
        val minFloor: Float = 0.0015f,
        /** Frames used to estimate the resting noise floor. */
        val noiseWindowFrames: Int = 24
    )

    enum class State {
        /** Waiting for the user to tap Start. */
        IDLE,
        /** Armed; watching for a sign to begin. */
        ARMED,
        /** A sign is in progress; frames are being buffered. */
        SIGNING,
        /** Sign finished; the segment is frozen and being classified. */
        PROCESSING
    }

    /** Why a sign ended, so the UI can explain a rejection honestly. */
    enum class EndReason {
        /** Quiet after motion — the ordinary case. */
        QUIET,
        /** Ran to [Config.maxSignMs] without going quiet. */
        TIMEOUT,
        /** No sign began before the app stopped waiting. */
        NO_SIGN,
        /** The app reached its deadline without a completed camera segment. */
        CAMERA_STALLED,
        /** Hand left the frame for longer than the grace period. */
        HAND_LOST,
        /** Segment was shorter than [Config.minSignMs]. */
        TOO_SHORT,
        /** User pressed Finish, which is always available as a fallback. */
        MANUAL
    }

    data class Outcome(
        val frames: List<HandTracker.Frame>,
        val reason: EndReason,
        val startedAtMs: Long,
        val endedAtMs: Long
    ) {
        val durationMs: Long get() = endedAtMs - startedAtMs
        val frameCount: Int get() = frames.size
        /** True when the segment is worth classifying. */
        val isUsable: Boolean get() = frames.isNotEmpty() &&
            (reason == EndReason.QUIET || reason == EndReason.HAND_LOST || reason == EndReason.MANUAL)
    }

    var state: State = State.IDLE
        private set

    /** Frames buffered for the in-progress sign, oldest first. */
    /**
     * Frames buffered for the in-progress sign.
     *
     * Capped even though [Config.maxSignMs] already bounds an attempt in time:
     * the timeout is a duration, and a device that delivers faster than expected
     * would otherwise grow this list in proportion to its frame rate. Dropping
     * the OLDEST frames past the cap is correct because only the model's 32
     * evenly-spaced samples survive anyway, and a very long attempt is already a
     * retry case.
     */
    private val buffer = ArrayList<HandTracker.Frame>(128)

    /** Recent motion samples, used to measure the resting noise floor. */
    private val recentMotion = ArrayDeque<Float>()

    private var armedAtMs = 0L
    private var handSeenSinceMs = 0L
    private var waitForHandRelease = false
    private var signingStartedAtMs = 0L
    private var lastMotionAtMs = 0L
    private var lastHandAtMs = 0L
    private var lastFrameAtMs = 0L
    private var previousFrame: HandTracker.Frame? = null
    private var preRollFrames = ArrayList<HandTracker.Frame>(32)

    /**
     * The resting noise floor, frozen at the instant the sign started.
     *
     * It is measured only while ARMED. Learning it during the sign would be
     * self-defeating: a signer moving steadily at 0.03 per frame would pull the
     * 25th percentile up to 0.03, making the floor 0.03 and the "moving"
     * threshold 0.054 — so the sign would immediately look quiet and end after
     * [Config.quietHoldMs]. Measured while the signer is waiting, the floor
     * describes their actual jitter instead of their signing.
     */
    private var frozenFloor: Float = 0f

    /** True while [state] is ARMED or SIGNING — i.e. the user has tapped Start. */
    val isActive: Boolean get() = state != State.IDLE && state != State.PROCESSING

    /** Frames captured so far in the current sign. */
    fun frameCount(): Int = buffer.size

    /** Arms the segmenter. Any previous sign is discarded. */
    fun arm(nowMs: Long, handAlreadyPresent: Boolean = false) {
        reset()
        armedAtMs = nowMs
        handSeenSinceMs = nowMs
        lastFrameAtMs = nowMs
        // Seeded to the arming instant, not 0: the "sustained motion" test
        // compares against it, and a zero would make the first moving frame
        // look arbitrarily old.
        lastMotionAtMs = nowMs
        waitForHandRelease = handAlreadyPresent
        state = State.ARMED
    }    /**
     * The user pressed Finish. Freezes whatever has been captured and reports
     * it as a manual end. Manual stop is always available and is the documented
     * fallback when auto-completion does not fire.
     */
    fun finishManually(nowMs: Long): Outcome = endSign(nowMs, EndReason.MANUAL)

    fun expire(nowMs: Long): Outcome {
        if (state == State.ARMED) {
            val outcome = Outcome(preRollFrames.toList(), EndReason.NO_SIGN, armedAtMs, nowMs)
            reset()
            return outcome
        }
        return endSign(nowMs, EndReason.CAMERA_STALLED)
    }

    /** Abandons the current attempt without producing a segment. */
    fun cancel() {
        reset()
    }

    private fun reset() {
        buffer.clear()
        preRollFrames.clear()
        recentMotion.clear()
        previousFrame = null
        armedAtMs = 0L
        signingStartedAtMs = 0L
        lastMotionAtMs = 0L
        lastHandAtMs = 0L
        waitForHandRelease = false
        lastFrameAtMs = 0L
        frozenFloor = 0f
        state = State.IDLE
    }

    /**
     * Feeds one analysed frame. Returns an [Outcome] on the frame that ends a
     * sign, and null on every other frame.
     *
     * Frames are always offered to the pre-roll ring first and only promoted
     * into the sign buffer once motion is confirmed, so the frames carrying the
     * initial handshape survive.
     */
    fun offer(frame: HandTracker.Frame): Outcome? {
        if (state != State.ARMED && state != State.SIGNING) return null

        val now = frame.capturedAtMs
        if (now < armedAtMs) return null
        val dtMs = if (lastFrameAtMs == 0L) 0L else (now - lastFrameAtMs).coerceAtLeast(0L)
        lastFrameAtMs = now

        val motion = motionSince(previousFrame, frame)
        previousFrame = frame

        val hasHand = frame.hands.isNotEmpty()
        if (hasHand) {
            lastHandAtMs = now
        }

        // Keep the pre-roll ring current: the frames just before onset are
        // exactly the ones carrying the initial handshape.
        preRollFrames.add(frame)
        trimPreRoll(now)

        return when (state) {
            State.ARMED -> considerOnset(frame, now, dtMs, motion, hasHand)
            State.SIGNING -> considerEnd(frame, now, motion, hasHand)
            else -> null
        }
    }

    private fun considerOnset(
        frame: HandTracker.Frame,
        now: Long,
        dtMs: Long,
        motion: Float,
        hasHand: Boolean
    ): Outcome? {
        if (now - armedAtMs < config.armingMs) {
            handSeenSinceMs = now
            return null
        }

        // Nothing to sign with.
        if (!hasHand) {
            // The hand left before a sign started; restart the acquisition
            // clock so re-entry gets a full settle period.
            handSeenSinceMs = now
            waitForHandRelease = false
            return null
        }
        if (now - handSeenSinceMs < config.acquireMs) return null

        // Learn the resting floor only here, while no sign is running.
        recordMotion(motion)
        val floor = noiseFloor()
        val heldForMs = now - handSeenSinceMs
        val moving = motion > floor * config.motionFloorMultiplier

        // Static signs: a hand held stable for a while is also a start. The bar
        // is lower because a held handshape legitimately has almost no motion.
        val heldStill = !waitForHandRelease && !moving && heldForMs >= config.poseHoldMs

        // Motion-based onset needs to be sustained, not a single noisy frame.
        val sustained = moving && now - lastMotionAtMs <= config.onsetConfirmMs

        if (!(sustained || heldStill)) {
            if (moving) lastMotionAtMs = now
            return null
        }

        state = State.SIGNING
        // The floor is frozen now, before any signing motion can bias it.
        frozenFloor = if (moving) floor else config.minFloor
        signingStartedAtMs = now - config.preRollMs
        lastMotionAtMs = now
        // Promote the pre-roll into the sign, including the current frame.
        buffer.clear()
        buffer.addAll(preRollFrames.filter { it.capturedAtMs >= signingStartedAtMs })
        if (buffer.none { it.capturedAtMs == frame.capturedAtMs }) buffer.add(frame)
        return null
    }

    private fun considerEnd(
        frame: HandTracker.Frame,
        now: Long,
        motion: Float,
        hasHand: Boolean
    ): Outcome? {
        buffer.add(frame)
        if (buffer.size > MAX_BUFFERED_FRAMES) buffer.removeAt(0)

        val floor = if (frozenFloor > 0f) frozenFloor else config.minFloor
        val moving = motion > floor * config.motionFloorMultiplier
        if (moving) {
            // Any real motion restarts the quiet timer. This is what lets a
            // two-part sign survive its internal pause.
            lastMotionAtMs = now
        }

        if (now - signingStartedAtMs >= config.maxSignMs) {
            return endSign(now, EndReason.TIMEOUT)
        }
        if (!hasHand && now - lastHandAtMs >= config.handLossGraceMs) {
            return endSign(now, EndReason.HAND_LOST)
        }
        // The quiet timer only runs while a hand is actually visible; a hand
        // leaving the frame must not be mistaken for a sign completing.
        if (hasHand && now - lastMotionAtMs >= config.quietHoldMs) {
            return endSign(now, EndReason.QUIET)
        }
        return null
    }

    private fun endSign(now: Long, reason: EndReason): Outcome {
        // A manual stop that arrives before any sign was detected has nothing
        // to classify. Reported as too short rather than as an empty segment
        // that later looks like a model failure.
        if (state == State.ARMED || buffer.isEmpty()) {
            val startedAt = if (state == State.ARMED) armedAtMs else signingStartedAtMs
            val outcome = Outcome(emptyList(), EndReason.TOO_SHORT, startedAt, now)
            reset()
            return outcome
        }
        // Trailing frames after the last motion belong to the sign's offset
        // (the hand returning to rest), so keep a short tail.
        val cutoff = if (reason == EndReason.QUIET) {
            (lastMotionAtMs + config.postRollMs).coerceAtMost(now)
        } else {
            now
        }
        val kept = buffer.filter { it.capturedAtMs in signingStartedAtMs..cutoff }
            .ifEmpty { buffer.toList() }
        val started = kept.firstOrNull()?.capturedAtMs ?: signingStartedAtMs
        val ended = kept.lastOrNull()?.capturedAtMs ?: now
        val effective = if (reason == EndReason.QUIET || reason == EndReason.MANUAL) {
            if (ended - started < config.minSignMs) EndReason.TOO_SHORT else reason
        } else {
            reason
        }
        val outcome = Outcome(kept, effective, started, ended)
        reset()
        return outcome
    }

    /**
     * Mean absolute landmark displacement between two frames, over the hands
     * present in BOTH. Zero when either side has no hand: a hand entering or
     * leaving is not motion of the sign.
     */
    private fun motionSince(prev: HandTracker.Frame?, cur: HandTracker.Frame): Float {
        if (prev == null || prev.hands.isEmpty() || cur.hands.isEmpty()) return 0f
        var sum = 0.0
        var count = 0
        for (hand in cur.hands) {
            val before = prev.hands.firstOrNull { it.slot == hand.slot } ?: continue
            for (j in 0 until minOf(before.xyz.size, hand.xyz.size)) {
                sum += kotlin.math.abs(hand.xyz[j] - before.xyz[j])
                count++
            }
        }
        return if (count == 0) 0f else (sum / count).toFloat()
    }

    private fun recordMotion(motion: Float) {
        recentMotion.addLast(motion)
        while (recentMotion.size > config.noiseWindowFrames) recentMotion.removeFirst()
    }

    /**
     * The signer's resting motion, as a low quantile of recent frames.
     *
     * A mean would be pulled up by the sign itself; a minimum would be set by a
     * single frozen frame. The 25th percentile tracks the quiet baseline while
     * staying robust to both.
     */
    private fun noiseFloor(): Float {
        if (recentMotion.isEmpty()) return config.initialFloor
        val sorted = recentMotion.sorted()
        val idx = (sorted.size - 1) / 4
        return sorted[idx].coerceAtLeast(config.minFloor)
    }

    private fun trimPreRoll(now: Long) {
        val cutoff = now - config.preRollMs
        while (preRollFrames.isNotEmpty() && preRollFrames.first().capturedAtMs < cutoff) {
            preRollFrames.removeAt(0)
        }
    }

    /** Snapshot of the frames captured so far, for the live UI counter. */
    fun snapshot(): List<HandTracker.Frame> = buffer.toList()
}
