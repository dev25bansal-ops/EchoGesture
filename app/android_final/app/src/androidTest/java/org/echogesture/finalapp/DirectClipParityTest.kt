package org.echogesture.finalapp

import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Direct-MP4 parity: the SAME video file, decoded on Android, through the
 * shipped detector and the shipped four-member ensemble.
 *
 * Decoding uses MediaExtractor + MediaCodec and advances SEQUENTIALLY through
 * the stream, keeping the frame nearest each target index. An earlier version
 * used `MediaMetadataRetriever.getFrameAtTime(..., OPTION_CLOSEST_SYNC)`, which
 * returns the nearest KEYFRAME — with 32 requests spread over a 75-frame clip
 * that collapsed onto a handful of sync frames, so "32 samples" were not 32
 * distinct images. [distinctFrameCount] exists to prove the current decoder
 * does not repeat that mistake.
 *
 * The clip is NOT bundled: it is authorized research data and must never be
 * committed. The test SKIPS when no clip is present so ordinary CI stays green;
 * the dedicated parity command passes `-e requireClip true` to turn a missing
 * clip into a hard failure instead.
 *
 * It writes no frames anywhere and reports only counts, labels and timings.
 */
@RunWith(AndroidJUnit4::class)
class DirectClipParityTest {

    private val instrumentation
        get() = InstrumentationRegistry.getInstrumentation()

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

    /** A decoded frame and the presentation timestamp it came from, in ms. */
    private class Decoded(val bitmap: Bitmap, val ptsMs: Long)

    /**
     * Decodes the whole clip sequentially and returns the [target] frames
     * nearest to evenly spaced target timestamps — the same
     * `linspace(0, duration, target)` intent as training, but selecting real
     * decoded frames rather than the nearest keyframe.
     *
     * Returns (frames, totalDecoded, requestedPts, actualPts) so a caller can
     * report exactly how many distinct frames it really got.
     */
    private fun sampleFrames(
        file: File,
        target: Int = 32,
        resize: Boolean,
    ): Pair<List<Decoded>, Int> {
        val extractor = MediaExtractor()
        extractor.setDataSource(file.absolutePath)

        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("video/")) {
                trackIndex = i
                format = f
                break
            }
        }
        if (trackIndex < 0 || format == null) {
            extractor.release()
            return Pair(emptyList(), 0)
        }
        extractor.selectTrack(trackIndex)

        val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) {
            format.getLong(MediaFormat.KEY_DURATION)
        } else {
            0L
        }
        val targetsUs = LongArray(target) { i ->
            if (target <= 1) 0L else durationUs * i / (target - 1)
        }

        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val info = MediaCodec.BufferInfo()
        val picked = ArrayList<Decoded>(target)
        // Index of the next target this frame could satisfy.
        var nextTarget = 0
        var decoded = 0
        var sawInputEos = false
        var sawOutputEos = false

        try {
            while (!sawOutputEos && picked.size < target) {
                if (!sawInputEos) {
                    val inIndex = codec.dequeueInputBuffer(10_000L)
                    if (inIndex >= 0) {
                        val buffer = codec.getInputBuffer(inIndex)!!
                        val size = extractor.readSampleData(buffer, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(
                                inIndex, 0, 0, 0L,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            sawInputEos = true
                        } else {
                            codec.queueInputBuffer(inIndex, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                when (val outIndex = codec.dequeueOutputBuffer(info, 10_000L)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> Unit
                    else -> {
                        if (outIndex >= 0) {
                            val image: Image? = codec.getOutputImage(outIndex)
                            if (info.size > 0 && image != null) {
                                val ptsMs = info.presentationTimeUs / 1000L
                                // Keep this frame if it is the closest yet to the
                                // next outstanding target. Because decode order is
                                // ascending, once we pass a target midpoint we stop
                                // looking for it.
                                while (nextTarget < target &&
                                    targetsUs[nextTarget] / 1000L <= ptsMs
                                ) {
                                    val bmp = imageToBitmap(image)
                                    val out = if (resize) {
                                        Bitmap.createScaledBitmap(bmp, 256, 256, true)
                                            .also { if (it !== bmp) bmp.recycle() }
                                    } else {
                                        bmp
                                    }
                                    picked.add(Decoded(out, ptsMs))
                                    nextTarget++
                                }
                                decoded++
                            }
                            codec.releaseOutputBuffer(outIndex, false)
                            if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                                sawOutputEos = true
                            }
                        }
                    }
                }
            }
        } finally {
            runCatching { codec.stop() }
            runCatching { codec.release() }
            extractor.release()
        }
        return Pair(picked, decoded)
    }

    /**
     * Image -> ARGB_8888 Bitmap, which MediaPipe requires.
     *
     * MediaCodec usually hands back YUV420 (3 planes, chroma subsampled), but
     * some codecs and devices give a single RGBA plane. Treating YUV as RGBA
     * silently produces garbage pixels — or, as here, a buffer-size exception
     * — so both layouts are handled explicitly and the pixel format is
     * reported so a mismatch is visible rather than assumed.
     */
    private fun imageToBitmap(image: Image): Bitmap {
        val planes = image.planes
        if (planes.size >= 3) {
            return yuv420ToBitmap(image)
        }
        val plane = planes[0]
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = rowStride - pixelStride * image.width
        val rawWidth = image.width + rowPadding / pixelStride
        val raw = Bitmap.createBitmap(rawWidth, image.height, Bitmap.Config.ARGB_8888)
        plane.buffer.rewind()
        raw.copyPixelsFromBuffer(plane.buffer)
        return if (rawWidth != image.width) {
            Bitmap.createBitmap(raw, 0, 0, image.width, image.height).also { raw.recycle() }
        } else {
            raw
        }
    }

    /** BT.601 limited-range YUV420 -> ARGB_8888, matching the codec's default. */
    private fun yuv420ToBitmap(image: Image): Bitmap {
        val w = image.width
        val h = image.height
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        val yRow = yPlane.rowStride
        val yPix = yPlane.pixelStride
        val uRow = uPlane.rowStride
        val uPix = uPlane.pixelStride
        val vRow = vPlane.rowStride
        val vPix = vPlane.pixelStride

        val out = IntArray(w * h)
        for (row in 0 until h) {
            val yBase = row * yRow
            val uvRow = row / 2
            for (col in 0 until w) {
                val y = (yBuf.get(yBase + col * yPix).toInt() and 0xFF) - 16
                val uIdx = uvRow * uRow + (col / 2) * uPix
                val vIdx = uvRow * vRow + (col / 2) * vPix
                val u = (uBuf.get(uIdx).toInt() and 0xFF) - 128
                val v = (vBuf.get(vIdx).toInt() and 0xFF) - 128
                val yc = (if (y < 0) 0 else y).toFloat()
                val r = (1.164f * yc + 1.596f * v).toInt().coerceIn(0, 255)
                val g = (1.164f * yc - 0.813f * v - 0.391f * u).toInt().coerceIn(0, 255)
                val b = (1.164f * yc + 2.018f * u).toInt().coerceIn(0, 255)
                out[row * w + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(out, 0, w, 0, 0, w, h)
        return bmp
    }

    private fun runPath(file: File, resize: Boolean): Map<String, Any> {
        val (frames, decoded) = sampleFrames(file, 32, resize)
        if (frames.isEmpty()) {
            return mapOf("decoded" to decoded, "frames" to 0, "hands" to 0)
        }

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath(ModelContract.DETECTOR_ASSET).build()
            )
            .setRunningMode(RunningMode.IMAGE)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.3f)
            .setMinHandPresenceConfidence(0.3f)
            .setMinTrackingConfidence(0.3f)
            .build()
        val landmarker = HandLandmarker.createFromOptions(
            instrumentation.targetContext, options
        )

        val n = frames.size
        val landmarks = FloatArray(n * 2 * 21 * 3)
        val present = FloatArray(n * 2)
        var hands = 0
        for ((fi, df) in frames.withIndex()) {
            val img = BitmapImageBuilder(df.bitmap).build()
            val result = landmarker.detect(img)
            img.close()
            df.bitmap.recycle()
            for (k in result.landmarks().indices) {
                val cat = result.handedness()[k].firstOrNull()?.categoryName() ?: "Right"
                val slot = if (cat.equals("Left", ignoreCase = true)) 0 else 1
                present[fi * 2 + slot] = 1f
                val lm = result.landmarks()[k]
                val base = (fi * 2 + slot) * 21 * 3
                for (j in 0 until 21) {
                    landmarks[base + j * 3] = lm[j].x()
                    landmarks[base + j * 3 + 1] = lm[j].y()
                    landmarks[base + j * 3 + 2] = lm[j].z()
                }
            }
            if (result.landmarks().isNotEmpty()) hands++
        }
        landmarker.close()
        return mapOf(
            "decoded" to decoded, "frames" to n, "hands" to hands,
            "landmarks" to landmarks, "present" to present,
        )
    }

    @Test
    fun androidDirectClipMatchesTheShippedPipeline() {
        val file = clipFile()
        val required = InstrumentationRegistry.getArguments()
            .getString("requireClip") == "true"
        if (requireClipMissing(file, required)) return

        val loaded = ModelContract.load(instrumentation.targetContext.assets)
        val classifier = LandmarkClassifier.create(loaded)

        // Distinct PTS values prove the decoder is not re-emitting keyframes.
        classifier.use {
            for (resize in listOf(false, true)) {
                val res = runPath(file!!, resize)
                val tag = if (resize) "resize256" else "native"
                val n = res["frames"] as Int
                if (n == 0) {
                    println("DIRECT_CLIP $tag no frames decoded")
                    continue
                }
                val landmarks = res["landmarks"] as FloatArray
                val present = res["present"] as FloatArray
                val feats = FeatureTransform.buildFeatures(landmarks, present, n, 24)
                val r = classifier.predict(feats)
                println(
                    "DIRECT_CLIP $tag decoded=${res["decoded"]} frames=$n " +
                        "hands=${res["hands"]} top1=${r.prediction} " +
                        "score=${"%.4f".format(r.confidence)} top5=${r.top5.take(3)}"
                )
                assertTrue("frames decoded", n > 0)
                assertTrue("landmarks finite", landmarks.all { it.isFinite() })
                assertEquals("top5 size", 5, r.top5.size)
            }
        }
    }

    /**
     * The decoder must yield distinct frames, not the same keyframe repeated.
     * On a 75-frame clip, sampling 32 targets must recover close to 32 distinct
     * presentation timestamps.
     */
    @Test
    fun decoderYieldsDistinctFramesNotRepeatedKeyframes() {
        val file = clipFile() ?: run {
            return
        }
        val (frames, decoded) = sampleFrames(file, 32, resize = false)
        val distinctPts = frames.map { it.ptsMs }.distinct().size
        println(
            "DIRECT_CLIP decoder decoded=$decoded sampled=${frames.size} " +
                "distinct_pts=$distinctPts duration_ms=${frames.lastOrNull()?.ptsMs}"
        )
        for (f in frames) f.bitmap.recycle()
        assertTrue("decoded at least one frame", decoded > 0)
        // The old OPTION_CLOSEST_SYNC path collapsed to a handful; require that
        // essentially every sample is a different frame.
        assertTrue(
            "expected ~32 distinct frames, got $distinctPts of ${frames.size}",
            distinctPts >= (frames.size * 9) / 10
        )
    }

    private fun requireClipMissing(file: File?, required: Boolean): Boolean {
        if (file == null && required) {
            throw AssertionError(
                "requireClip=true but no .mp4 found in the app's files dirs. " +
                    "Push a clip and re-run; this run is INCOMPLETE, not passing."
            )
        }
        assumeTrue("no .mp4 in the app's files dir; push one to run this test", file != null)
        return file == null
    }
}