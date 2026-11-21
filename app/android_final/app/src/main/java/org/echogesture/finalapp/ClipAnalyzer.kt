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

    /**
     * Hard cap on decoded frames.
     *
     * This is a MEMORY bound, and the previous value was not one: 3600 frames of
     * 640x480 ARGB_8888 is 3600 * 1.17 MiB = ~4.2 GiB, which no phone holds. The
     * comment claimed it "bounds memory at ~2 min of 30 fps" while counting
     * frames and not bytes. Every frame is decoded before any is detected, so a
     * 60 s gallery clip would have reached ~2.1 GiB and been OOM-killed
     * mid-analysis with no result card.
     *
     * 120 is chosen from the work the classifier actually needs: the training
     * path samples 32 frames across a clip, and the longest real clip measured
     * in this project is 74 frames. 120 keeps a 60% margin over that while
     * bounding peak decode memory to 120 * 1.17 MiB = ~140 MiB, which is a size
     * a phone can actually hold. Longer clips are truncated, and `truncated` is
     * reported so the caller can tell the user.
     */
    private const val MAX_FRAMES_DEFAULT = 120

    /**
     * Longest edge fed to MediaPipe. The camera path delivers ~640x480, and
     * the detector normalises coordinates, so an arbitrary-resolution upload is
     * scaled here to keep the detector in the regime it was measured in rather
     * than silently feeding it 4K.
     */
    private const val MAX_EDGE = 640

    /**
     * Fewest frames the sequential decoder must produce before its output is
     * trusted over the retriever's.
     *
     * A codec that starts but dribbles frames out (unsupported profile, a
     * truncated container) would otherwise silently replace a complete clip
     * with a handful of images. Below this, the retriever is used instead.
     */
    private const val MIN_SEQUENTIAL_FRAMES = 10

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
        // Sequential codec first, per-frame seeking as the fallback.
        //
        // The retriever seeks to a keyframe and decodes forward for EVERY frame
        // requested, which cost 19,872 ms of a 22,710 ms run for 55 frames
        // before scaling, and is still 8,698 ms after it. One pass with
        // MediaExtractor+MediaCodec decodes each frame exactly once.
        //
        // Falling back rather than failing is deliberate: a codec can refuse a
        // container or a profile the retriever handles, and a slower correct
        // answer beats a fast error.
        var decodeNote = ""
        val frames = try {
            // Sequential codec FIRST, retriever as fallback.
            //
            // The retriever seeks to a keyframe and decodes forward for EVERY
            // requested frame: measured 8,700-21,200 ms for a 55-frame clip.
            // MediaExtractor + MediaCodec decodes the stream once.
            //
            // This path is only used because it was proven on this device
            // first: SequentialDecoderSmokeTest decodes a real clip in ~870 ms
            // for 23 frames (a 10-24x speedup), with a luma-spread assertion
            // proving the YUV conversion produces a real image rather than a
            // blank bitmap. An earlier surface-based attempt aborted natively in
            // ImageReader$SurfaceImage.nativeCreatePlanes and killed the app, so
            // the smoke test exists specifically to catch that class of failure
            // before users do.
            //
            // Output is in decoder order, which for these containers is
            // presentation order; ClipAnalyzer resamples by index downstream, so
            // order only has to be temporal, which it is.
            val seq = runCatching {
                SequentialVideoDecoder.decode(
                    context, uri, targetFps = 30, maxEdge = MAX_EDGE, maxFrames = maxFrames
                )
            }.getOrNull()
            if (seq != null && seq.bitmaps.size >= MIN_SEQUENTIAL_FRAMES) {
                decodeNote = "sequential codec=${seq.codecName} decode=${seq.decodeMs}ms"
                seq.bitmaps.mapIndexed { i, b ->
                    Decoded(b, seq.ptsMs.getOrElse(i) { 0L })
                }
            } else {
                decodeNote = "retriever" + (seq?.let {
                    " (sequential gave ${it.bitmaps.size} frames)"
                } ?: " (codec unavailable)")
                // The sequential path may have produced a few frames before
                // giving up; release them before falling back so the retriever's
                // copies are not competing with them for memory.
                seq?.bitmaps?.forEach { runCatching { it.recycle() } }
                decode(context, uri, maxFrames)
            }
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
                val hands = toHands(result, copy.width, copy.height)
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
            "clip stages: $decodeNote decode=${tDecoded - tStart}ms " +
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
            // A square request is correct for BOTH orientations: the retriever
            // returns the frame already rotated upright, and getScaledFrameAtTime
            // preserves aspect ratio (it scales by a single uniform factor), so
            // asking for 640x640 yields 640x480 landscape or 480x640 portrait
            // without stretching either. The old orientation-dependent dims
            // existed only to compensate for the double-rotation that is now
            // removed.
            val targetW = MAX_EDGE
            val targetH = MAX_EDGE
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
    /**
     * Fit the longest edge to [MAX_EDGE]. Rotation is NOT applied here.
     *
     * `MediaMetadataRetriever` already rotates: its JNI allocates the output
     * bitmap with width and height swapped for 90/270 and copies pixels through
     * rotate90()/rotate270() using the container's rotation metadata. That has
     * been true from android-4.4_r1 through main, covering minSdk 24. Applying
     * `postRotate(rotationDegrees)` on top would rotate a portrait clip a SECOND
     * time and hand the detector a sideways frame.
     *
     * Every clip used in this project's measurements is 640x480 with rotation 0,
     * so this was a no-op on all of them and no result in the repo changes.
     * It would only have appeared on a portrait phone recording. That case has
     * no test clip yet, which is why the rotation is removed (strictly safer
     * than applying it twice) rather than replaced with new logic.
     */
    private fun scaleAndRotate(src: Bitmap, rotationDegrees: Int): Bitmap {
        val w = src.width
        val h = src.height
        val longest = maxOf(w, h)
        val scale = if (longest > MAX_EDGE) MAX_EDGE.toFloat() / longest else 1f
        if (scale == 1f) return src
        val matrix = android.graphics.Matrix()
        matrix.postScale(scale, scale)
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
    private fun toHands(
        result: com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult,
        width: Int,
        height: Int,
    ): List<HandTracker.Hand> {
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
            // A decoded clip carries its own size, which need not be the 4:3
            // the checkpoints were trained on. This is the DECODED FRAME's
            // size -- the same "buffer" the detector was handed -- because
            // MediaPipe reports coordinates in the frame of the image it was
            // given, never in a rotated display frame. The camera path passes
            // the identical dimensions for the identical reason, so a sign
            // classifies the same way whichever route it arrives by.
            HandTracker.mapToCorpusAspect(arr, width, height)
            out.add(HandTracker.Hand(slot, arr))
            i++
        }
        return out
    }
}
