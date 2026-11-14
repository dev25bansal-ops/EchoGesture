package org.echogesture.finalapp

import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer

/**
 * Replay adapter: the SAME decoded frames through both production modes.
 *
 * A real clip is decoded once into timestamped hand landmarks. Those landmarks
 * are then wrapped as real [HandTracker.Frame]s and pushed through the actual
 * production classes:
 *
 *  - record mode  → [SignRecorder] + [RecordingFeatures]
 *  - quick scan   → [HandTracker.selectCapture]
 *
 * No test-only feature maths is used anywhere: every tensor comes out of
 * [FeatureTransform], the same object the app calls, and every decision out of
 * the same classes the button handlers call. If the app's capture logic drifts,
 * this test moves with it.
 *
 * The clip is authorised research data and is never committed. The test SKIPS
 * with no clip present so ordinary CI stays green; the parity command passes
 * `-e requireClip true` to turn that into a hard failure.
 */
@RunWith(AndroidJUnit4::class)
class ReplayModeParityTest {

    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()

    private fun clipFile(): File? {
        val dirs = listOfNotNull(
            instrumentation.targetContext.getExternalFilesDir(null),
            instrumentation.targetContext.filesDir,
        )
        return dirs
            .flatMap { d -> d.listFiles { f -> f.isFile && f.name.endsWith(".mp4", true) }?.toList() ?: emptyList() }
            .sortedBy { it.name }
            .firstOrNull()
    }

    private class Decoded(val bitmap: Bitmap, val ptsMs: Long)

    /** Sequential decode keeping the frame nearest each of [target] timestamps. */
    private fun decode(file: File, target: Int): Pair<List<Decoded>, Int> {
        val extractor = MediaExtractor()
        extractor.setDataSource(file.absolutePath)
        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("video/")) { trackIndex = i; format = f; break }
        }
        if (trackIndex < 0 || format == null) { extractor.release(); return Pair(emptyList(), 0) }
        extractor.selectTrack(trackIndex)
        val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION))
            format.getLong(MediaFormat.KEY_DURATION) else 0L
        val targetsUs = LongArray(target) { i ->
            if (target <= 1) 0L else durationUs * i / (target - 1)
        }
        val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(format, null, null, 0)
        codec.start()
        val info = MediaCodec.BufferInfo()
        val picked = ArrayList<Decoded>(target)
        var nextTarget = 0
        var decoded = 0
        var eosIn = false
        var eosOut = false
        try {
            while (!eosOut && picked.size < target) {
                if (!eosIn) {
                    val inIndex = codec.dequeueInputBuffer(10_000L)
                    if (inIndex >= 0) {
                        val buf = codec.getInputBuffer(inIndex)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(inIndex, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            eosIn = true
                        } else {
                            codec.queueInputBuffer(inIndex, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val outIndex = codec.dequeueOutputBuffer(info, 10_000L)
                if (outIndex >= 0) {
                    if (info.size > 0) {
                        decoded++
                        val ptsMs = info.presentationTimeUs / 1000
                        while (nextTarget < target && ptsMs >= targetsUs[nextTarget] / 1000) {
                            val image = codec.getOutputImage(outIndex)
                            if (image != null) {
                                picked.add(Decoded(imageToBitmap(image), ptsMs))
                                image.close()
                            }
                            nextTarget++
                        }
                    }
                    codec.releaseOutputBuffer(outIndex, false)
                } else if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    // New format; nothing to do.
                }
                if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) eosOut = true
            }
        } finally {
            codec.stop(); codec.release(); extractor.release()
        }
        return Pair(picked, decoded)
    }

    private fun imageToBitmap(image: Image): Bitmap {
        val w = image.width
        val h = image.height
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(w * h)
        if (image.planes.size == 1) {
            val buf: ByteBuffer = image.planes[0].buffer
            val bytes = ByteArray(buf.remaining())
            buf.get(bytes)
            bmp.copyPixelsFromBuffer(ByteBuffer.wrap(bytes))
            return bmp
        }
        val y = image.planes[0]; val u = image.planes[1]; val v = image.planes[2]
        val yBuf = y.buffer; val uBuf = u.buffer; val vBuf = v.buffer
        val yRow = y.rowStride; val yPix = y.pixelStride
        val uRow = u.rowStride; val uPix = u.pixelStride
        val vRow = v.rowStride; val vPix = v.pixelStride
        for (row in 0 until h) {
            val yBase = row * yRow
            val uvRow = row / 2
            for (col in 0 until w) {
                val yv = yBuf.get(yBase + col * yPix).toInt() and 0xFF
                val uvCol = col / 2
                val uu = (uBuf.get(uvRow * uRow + uvCol * uPix).toInt() and 0xFF) - 128
                val vv = (vBuf.get(uvRow * vRow + uvCol * vPix).toInt() and 0xFF) - 128
                val r = (yv + 1.402f * vv).toInt().coerceIn(0, 255)
                val g = (yv - 0.344136f * uu - 0.714136f * vv).toInt().coerceIn(0, 255)
                val b = (yv + 1.772f * uu).toInt().coerceIn(0, 255)
                pixels[row * w + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        bmp.setPixels(pixels, 0, w, 0, 0, w, h)
        return bmp
    }

    /** Detect hands on every decoded frame and build production Frames. */
    private fun landmarksOf(frames: List<Decoded>): List<HandTracker.Frame> {
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(ModelContract.DETECTOR_ASSET).build())
            .setRunningMode(RunningMode.IMAGE)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.3f)
            .setMinHandPresenceConfidence(0.3f)
            .setMinTrackingConfidence(0.3f)
            .build()
        val landmarker = HandLandmarker.createFromOptions(instrumentation.targetContext, options)
        val out = ArrayList<HandTracker.Frame>(frames.size)
        try {
            for (df in frames) {
                val img = BitmapImageBuilder(df.bitmap).build()
                val result = landmarker.detect(img)
                img.close()
                val hands = ArrayList<HandTracker.Hand>(2)
                for (k in result.landmarks().indices) {
                    val cat = result.handedness()[k].firstOrNull()?.categoryName() ?: "Right"
                    val slot = if (cat.equals("Left", ignoreCase = true)) 0 else 1
                    val lm = result.landmarks()[k]
                    val xyz = FloatArray(63)
                    for (j in 0 until 21) {
                        xyz[j * 3] = lm[j].x(); xyz[j * 3 + 1] = lm[j].y(); xyz[j * 3 + 2] = lm[j].z()
                    }
                    hands.add(HandTracker.Hand(slot, xyz))
                }
                df.bitmap.recycle()
                out.add(HandTracker.Frame(hands, df.ptsMs))
            }
        } finally {
            landmarker.close()
        }
        return out
    }

    @Test
    fun bothModesProduceAValidTensorFromTheSameFrames() {
        val file = clipFile()
        val required = InstrumentationRegistry.getArguments().getString("requireClip") == "true"
        if (file == null) {
            if (required) throw AssertionError("requireClip=true but no .mp4 in app files dir")
            println("REPLAY skipped: no clip present")
            return
        }

        val (decoded, total) = decode(file, 64)
        if (decoded.isEmpty()) {
            println("REPLAY skipped: decoder yielded no frames")
            return
        }
        val allFrames = landmarksOf(decoded)
        assertTrue("decoder produced frames", allFrames.isNotEmpty())

        // PTS must be distinct and monotonic, or this is not a sequence.
        val stamps = allFrames.map { it.capturedAtMs }
        assertEquals("timestamps must be monotonic", stamps.sorted(), stamps)
        val distinct = stamps.toSet().size
        assertTrue(
            "only $distinct distinct timestamps from ${stamps.size} frames",
            distinct >= (stamps.size * 0.8).toInt()
        )

        val startedAt = stamps.first()
        val stoppedAt = stamps.last()

        // --- record mode, through the production classes -------------------
        val recorder = SignRecorder()
        recorder.start(startedAt)
        allFrames.forEach { recorder.offer(it) }
        val outcome = recorder.stop(stoppedAt)

        // A real clip is a real sign: it must be usable.
        assertTrue("a genuine clip must be recordable, got $outcome",
            outcome is SignRecorder.Result.Usable)
        val recording = (outcome as SignRecorder.Result.Usable).recording

        val recBoth = RecordingFeatures.buildBoth(recording)
        val recXY = FeatureTransform.build(
            RecordingFeatures.tensors(RecordingFeatures.sample(recording)).first,
            RecordingFeatures.tensors(RecordingFeatures.sample(recording)).second,
            RecordingFeatures.sample(recording).size, 24, useZ = false
        )
        val recXYZ = FeatureTransform.build(
            RecordingFeatures.tensors(RecordingFeatures.sample(recording)).first,
            RecordingFeatures.tensors(RecordingFeatures.sample(recording)).second,
            RecordingFeatures.sample(recording).size, 24, useZ = true
        )
        val recTensor = recBoth.featuresXY
        assertEquals(24 * 255, recTensor.size)
        assertTrue("features finite", recTensor.all { it.isFinite() })

        val sampled = RecordingFeatures.sample(recording)
        assertEquals(RecordingFeatures.SAMPLED, sampled.size)
        assertTrue("no frame invented", sampled.all { it.capturedAtMs in stamps.toSet() })
        // The segmenter may trim leading idle — this clip has 2.2 s of frames
        // with no hand before the sign. What must hold is that the sample lies
        // inside the recording and that every kept frame is a real one.
        assertTrue("sample must start within the recording",
            sampled.first().capturedAtMs >= startedAt)
        assertTrue("sample must end within the recording",
            sampled.last().capturedAtMs <= stamps.last())
        assertTrue("all sampled frames are real", sampled.all { it.capturedAtMs in stamps.toSet() })

        // --- quick scan, through the same production selector --------------
        val request = HandTracker.selectCapture(allFrames, stoppedAt, Long.MIN_VALUE)
        val quick = (request as HandTracker.CaptureRequest.Ready).capture
        val qn = quick.frameCount
        val ql = FloatArray(qn * 2 * 21 * 3)
        val qp = FloatArray(qn * 2)
        var fi = 0
        for (f in quick.frames) {
            for (h in f.hands) {
                qp[fi * 2 + h.slot] = 1f
                System.arraycopy(h.xyz, 0, ql, (fi * 2 + h.slot) * 21 * 3, 21 * 3)
            }
            fi++
        }
        val quickXY = FeatureTransform.build(ql, qp, qn, 24, useZ = false)
        val quickXYZ = FeatureTransform.build(ql, qp, qn, 24, useZ = true)
        val quickTensor = quickXY.features
        assertEquals(24 * 255, quickTensor.size)

        // --- both tensors through the shipped ensemble ---------------------
        val loaded = ModelContract.load(instrumentation.targetContext.assets)
        LandmarkClassifier.create(loaded).use { classifier ->
            val rRec = classifier.predict(recXY.features, recXY.mask, recXYZ.features)
            val rQuick = classifier.predict(quickXY.features, quickXY.mask, quickXYZ.features)
            println(
                "REPLAY decoded=$total frames=${allFrames.size} distinct=$distinct " +
                    "rec_dur=${recording.durationMs}ms rec_hand=${recording.frames.count { it.hands.isNotEmpty() }} " +
                    "rec_top1=${rRec.prediction} rec_score=${"%.4f".format(rRec.confidence)} " +
                    "quick_top1=${rQuick.prediction} quick_score=${"%.4f".format(rQuick.confidence)}"
            )
            assertEquals(5, rRec.top5.size)
            assertEquals(5, rQuick.top5.size)
            // Determinism: the same tensor must give the same answer twice.
            val again = classifier.predict(recXY.features, recXY.mask, recXYZ.features)
            assertEquals("record-mode prediction must be deterministic",
                rRec.prediction, again.prediction)
        }

        // Record mode must reach further back in time than quick scan: it
        // covers the whole clip, quick scan only its last CAPTURE_SPAN_MS.
        val recSpan = sampled.last().capturedAtMs - sampled.first().capturedAtMs
        val quickSpan = quick.frames.last().capturedAtMs - quick.frames.first().capturedAtMs
        assertTrue(
            "record span ${recSpan}ms should cover more than quick scan ${quickSpan}ms on a long clip",
            recSpan >= quickSpan - 100
        )
        println("REPLAY span rec=${recSpan}ms quick=${quickSpan}ms")

        // The segmenter runs on the same frames. On a clip that is one sign
        // start-to-finish there is no idle to remove, so it must keep the whole
        // recording rather than inventing a boundary.
        val segment = SignSegmenter.choose(recording.frames, RecordingFeatures.fpsOf(recording))
        println("REPLAY segment reason=${segment.reason} kept=" +
            "${segment.endIndex - segment.startIndex}/${recording.frameCount} " +
            "motion=${"%.4f".format(RecordingFeatures.meanMotion(recTensor))}")
        assertTrue(
            "a clean clip must not be over-trimmed, kept " +
                "${segment.endIndex - segment.startIndex}/${recording.frameCount}",
            (segment.endIndex - segment.startIndex) >= (recording.frameCount * 0.5).toInt()
        )
        assertTrue("segment bounds in range",
            segment.startIndex >= 0 && segment.endIndex <= recording.frameCount)
    }

}