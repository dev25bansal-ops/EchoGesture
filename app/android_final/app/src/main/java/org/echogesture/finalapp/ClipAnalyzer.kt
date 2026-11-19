package org.echogesture.finalapp

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker

/**
 * Turns a video FILE into the same frame sequence the camera would have
 * produced, so a clip can be classified by the identical [RecordingFeatures]
 * → [LandmarkClassifier] path.
 *
 * Why this exists
 * ---------------
 * Measured on 2026-11-19: the NOON clip scores NOON1 at 0.9731 through the
 * Python pipeline and again through the Android classifier, but a live camera
 * capture of the same sign returned FOREIGNER1 at 0.51-0.60 with NOON1 ranked
 * 2nd. Temporal subsampling was ruled out as the cause (at 7.5 and 15 fps the
 * clip still scores NOON1 at 0.955-0.963), which leaves the camera's optics
 * and the scene in front of the lens. A file never has those: the pixels are
 * the ones the model was measured on. So this path is both a real feature and
 * the deterministic test the phone could not give us before.
 *
 * Contract
 * --------
 * Landmarks come from a [HandLandmarker] built with the SAME options as
 * [HandTracker] (2 hands, 0.3 thresholds, VIDEO mode, GPU with CPU fallback).
 * Frames are timestamped from the clip's own presentation times, which keeps
 * the real frame rate — [RecordingFeatures.fpsOf] and [SignSegmenter] both
 * depend on it. Frames with no hand carry an empty list, exactly as the camera
 * path does, so a hand-less section keeps its temporal position.
 *
 * Deliberately NOT limited to 32 frames: the same 32->24 resample runs
 * downstream in [RecordingFeatures], and giving it the whole clip is what
 * `read_frames()` does in training.
 */
object ClipAnalyzer {

    /** Hard cap on decoded frames; bounds memory at ~2 min of 30 fps. */
    private const val MAX_FRAMES_DEFAULT = 3600

    /**
     * Longest edge fed to MediaPipe. The camera path delivers ~640x480, and
     * the detector normalises coordinates, so an arbitrary-resolution upload is
     * scaled here to keep the detector in the regime it was measured in rather
     * than silently feeding it 4K.
     */
    private const val MAX_EDGE = 640

    sealed interface Outcome {
        data class Analysed(
            val recording: SignRecorder.Recording,
            val frameCount: Int,
            val handFrames: Int,
            /** True when [MAX_FRAMES_DEFAULT] cut the clip short. */
            val truncated: Boolean
        ) : Outcome

        data class Failed(val reason: Reason, val detail: String = "") : Outcome {
            enum class Reason {
                /** The file could not be opened or decoded as video. */
                UNREADABLE,
                /** Opened fine but produced no frames. */
                NO_FRAMES,
                /** The detector could not be created. */
                DETECTOR_UNAVAILABLE,
            }
        }
    }

    /**
     * Decodes and analyses [uri] synchronously. Call from a worker thread; this
     * blocks for the length of the clip and MediaPipe will initialise on first
     * use (~1-2 min on the slowest measured device, tens of ms once warm).
     */
    fun analyse(context: Context, uri: Uri, maxFrames: Int = MAX_FRAMES_DEFAULT): Outcome {
        val tStart = android.os.SystemClock.elapsedRealtime()
        val frames = try {
            decode(context, uri, maxFrames)
        } catch (t: Throwable) {
            return Outcome.Failed(Outcome.Failed.Reason.UNREADABLE, t.javaClass.simpleName)
        }
        if (frames.isEmpty()) {
            return Outcome.Failed(Outcome.Failed.Reason.NO_FRAMES)
        }
        val tDecoded = android.os.SystemClock.elapsedRealtime()

        val landmarker = try {
            createLandmarker(context)
        } catch (t: Throwable) {
            for (f in frames) runCatching { f.bitmap.recycle() }
            return Outcome.Failed(
                Outcome.Failed.Reason.DETECTOR_UNAVAILABLE,
                t.javaClass.simpleName
            )
        }
        val tLandmarker = android.os.SystemClock.elapsedRealtime()

        val out = ArrayList<HandTracker.Frame>(frames.size)
        var handFrames = 0
        var copyMs = 0L
        var detectMs = 0L
        var toHandsMs = 0L
        // VIDEO mode requires strictly increasing timestamps. Container PTS is
        // not guaranteed monotonic after scaling (two targets can resolve to
        // the same millisecond) and MediaPipe faults on a repeat, so clamp.
        var lastTs = Long.MIN_VALUE / 2
        try {
            for (f in frames) {
                val c0 = android.os.SystemClock.elapsedRealtime()
                val copy = f.bitmap.copy(Bitmap.Config.ARGB_8888, false)
                val img = BitmapImageBuilder(copy).build()
                val c1 = android.os.SystemClock.elapsedRealtime()
                val ts = maxOf(f.ptsMs, lastTs + 1L)
                lastTs = ts
                val result = landmarker.detectForVideo(img, ts)
                val c2 = android.os.SystemClock.elapsedRealtime()
                img.close()
                val hands = toHands(result)
                val c3 = android.os.SystemClock.elapsedRealtime()
                copyMs += c1 - c0
                detectMs += c2 - c1
                toHandsMs += c3 - c2
                if (hands.isNotEmpty()) handFrames++
                out.add(HandTracker.Frame(hands, f.ptsMs))
            }
        } catch (t: Throwable) {
            return Outcome.Failed(Outcome.Failed.Reason.UNREADABLE, t.javaClass.simpleName)
        } finally {
            landmarker.close()
            for (f in frames) runCatching { f.bitmap.recycle() }
        }

        if (out.size < 2) {
            return Outcome.Failed(Outcome.Failed.Reason.NO_FRAMES)
        }
        // Per-stage breakdown. A single total tells you the clip path is slow
        // but not which half to fix, and the two halves have opposite fixes:
        // slow DECODE means MediaMetadataRetriever, slow DETECT means the
        // delegate is not being used.
        val tEnd = android.os.SystemClock.elapsedRealtime()
        android.util.Log.i(
            "EchoGesture",
            "clip stages: decode=${tDecoded - tStart}ms " +
                "landmarkerInit=${tLandmarker - tDecoded}ms " +
                "copy=${copyMs}ms detect=${detectMs}ms toHands=${toHandsMs}ms " +
                "total=${tEnd - tStart}ms frames=${out.size} " +
                "perFrame=${(detectMs / out.size.coerceAtLeast(1))}ms"
        )
        val startedAt = out.first().capturedAtMs
        val stoppedAt = out.last().capturedAtMs
        return Outcome.Analysed(
            recording = SignRecorder.Recording(out, startedAt, stoppedAt),
            frameCount = out.size,
            handFrames = handFrames,
            truncated = frames.size >= maxFrames
        )
    }

    /** One decoded frame with the presentation time it will be handed to MediaPipe at. */
    private class Decoded(val bitmap: Bitmap, val ptsMs: Long)

    private fun decode(context: Context, uri: Uri, maxFrames: Int): List<Decoded> {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            val durationMs = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L
            val rotation = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                ?.toIntOrNull() ?: 0

            // Ask for frames at the clip's own rate. MediaMetadataRetriever has
            // no frame iterator, so the rate is derived from the clip duration
            // and a fixed target of ~30 fps — the rate the training clips use.
            // getFrameAtTime() snaps to the nearest keyframe-aware time, so the
            // real returned timestamps are what the recording is built from.
            val targetFps = 30
            val stepUs = 1_000_000L / targetFps
            val totalUs = durationMs * 1000L
            if (totalUs <= 0) return emptyList()

            val out = ArrayList<Decoded>()
            var tUs = 0L
            var lastPts = -1L
            // Decode-time downscale. getScaledFrameAtTime() asks the codec for a
            // frame already at the target size, so the full-resolution bitmap is
            // never allocated and never thrown away. Measured on the Vivo V2404
            // (2026-11-20): decoding 55 frames at source size cost 19,872 ms of
            // a 22,710 ms run -- 87% of the wall time, and every one of those
            // pixels except ~4% was discarded by scaleAndRotate afterwards.
            //
            // The scaling happens inside the decoder, so it is close to free
            // compared with decoding full size and resampling on the CPU.
            val targetW = if (rotation == 90 || rotation == 270) MAX_EDGE * 3 / 4 else MAX_EDGE
            val targetH = if (rotation == 90 || rotation == 270) MAX_EDGE else MAX_EDGE * 3 / 4
            while (tUs < totalUs && out.size < maxFrames) {
                // OPTION_CLOSEST, not OPTION_CLOSEST_SYNC.
                //
                // CLOSEST_SYNC is ~2.3x faster but returns only KEYFRAMES, so a
                // 55-frame request yielded 54 near-duplicate images with no
                // detectable hand and the app reported "No hands in view" for a
                // clip it had just classified correctly. Measured on device:
                // decode 19,872ms -> 8,698ms, and the answer went from NOON1 to
                // nothing. A fast wrong answer is worse than a slow right one.
                //
                // The scaling below is the safe speedup: it cuts decode work
                // without changing WHICH frame is returned.
                val bmp = runCatching {
                    retriever.getScaledFrameAtTime(
                        tUs,
                        MediaMetadataRetriever.OPTION_CLOSEST,
                        targetW,
                        targetH
                    )
                }.getOrNull()
                    ?: runCatching {
                        retriever.getFrameAtTime(tUs, MediaMetadataRetriever.OPTION_CLOSEST)
                    }.getOrNull()
                    ?: break
                val scaled = scaleAndRotate(bmp, rotation)
                if (scaled !== bmp) runCatching { bmp.recycle() }
                // A repeated timestamp means the retriever snapped to the same
                // source frame; skip it rather than pad the sequence with a
                // duplicate. The +1 is applied at detect time, not here, so the
                // recording keeps the clip's true timing.
                val ptsMs = tUs / 1000L
                if (ptsMs != lastPts) {
                    out.add(Decoded(scaled, ptsMs))
                    lastPts = ptsMs
                } else {
                    runCatching { scaled.recycle() }
                }
                tUs += stepUs
            }
            return out
        } finally {
            runCatching { retriever.release() }
        }
    }

    /** Fit the longest edge to [MAX_EDGE] and bake in the container rotation. */
    private fun scaleAndRotate(src: Bitmap, rotationDegrees: Int): Bitmap {
        val w = src.width
        val h = src.height
        val longest = maxOf(w, h)
        val scale = if (longest > MAX_EDGE) MAX_EDGE.toFloat() / longest else 1f
        val matrix = android.graphics.Matrix()
        if (scale != 1f) matrix.postScale(scale, scale)
        if (rotationDegrees != 0) matrix.postRotate(rotationDegrees.toFloat())
        if (matrix.isIdentity) return src
        val out = Bitmap.createBitmap(src, 0, 0, w, h, matrix, true)
        return if (out === src) src else out
    }

    private fun createLandmarker(context: Context): HandLandmarker {
        fun build(delegate: Delegate): HandLandmarker =
            HandLandmarker.createFromOptions(
                context,
                HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(
                        BaseOptions.builder()
                            .setModelAssetPath(ModelContract.DETECTOR_ASSET)
                            .setDelegate(delegate)
                            .build()
                    )
                    .setRunningMode(RunningMode.VIDEO)
                    .setNumHands(2)
                    .setMinHandDetectionConfidence(0.3f)
                    .setMinHandPresenceConfidence(0.3f)
                    .setMinTrackingConfidence(0.3f)
                    .build()
            )
        return try {
            build(Delegate.GPU)
        } catch (t: Throwable) {
            build(Delegate.CPU)
        }
    }

    /** Identical handedness mapping to [HandTracker]: Left=0, Right=1, no mirroring. */
    private fun toHands(result: com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult):
        List<HandTracker.Hand> {
        val out = ArrayList<HandTracker.Hand>(result.landmarks().size)
        var i = 0
        for (pts in result.landmarks()) {
            val handedness = if (i < result.handedness().size && result.handedness()[i].isNotEmpty())
                result.handedness()[i][0].categoryName()
            else "Right"
            val slot = if (handedness.equals("Left", ignoreCase = true)) 0 else 1
            val arr = FloatArray(21 * 3)
            for (j in 0 until 21) {
                arr[j * 3] = pts[j].x()
                arr[j * 3 + 1] = pts[j].y()
                arr[j * 3 + 2] = pts[j].z()
            }
            out.add(HandTracker.Hand(slot, arr))
            i++
        }
        return out
    }
}
