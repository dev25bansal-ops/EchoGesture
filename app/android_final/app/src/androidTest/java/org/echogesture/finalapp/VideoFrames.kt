package org.echogesture.finalapp

import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.File

/**
 * Decode a video file into evenly-spaced Bitmaps for instrumentation tests.
 *
 * Extracted from three near-identical private copies in ClipReplayAccuracyTest,
 * DirectClipParityTest and ReplayModeParityTest. Consolidating it fixed a real
 * defect along the way: the originals looped on
 * `while (!sawOutputEos && picked.size < target)` with no bound, so a codec that
 * never emitted EOS — or never emitted an output buffer at all — spun forever.
 * That is the most likely cause of
 * `androidPipelineAccuratelyClassifiesLabelledClips` being killed at a
 * 30-minute timeout having printed nothing.
 *
 * [maxIterations] bounds the loop so a stuck codec becomes a clean, reported
 * failure instead of a hang. Callers get whatever frames were recovered plus a
 * flag saying whether the decode completed.
 */
object VideoFrames {

    data class Decoded(val bitmap: Bitmap, val ptsMs: Long)

    data class Result(
        val frames: List<Decoded>,
        /** How many frames the codec actually produced (not how many were kept). */
        val decodedCount: Int,
        /** False when the iteration cap or a codec error ended the loop early. */
        val completed: Boolean,
    )

    /** Generous but finite: ~2 s of wall-clock at 10 ms per dequeue attempt. */
    private const val DEFAULT_MAX_ITERATIONS = 20_000

    fun sample(
        file: File,
        target: Int,
        resize: Boolean = false,
        maxIterations: Int = DEFAULT_MAX_ITERATIONS,
    ): Result {
        val extractor = MediaExtractor()
        extractor.setDataSource(file.absolutePath)
        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            if ((f.getString(MediaFormat.KEY_MIME) ?: "").startsWith("video/")) {
                trackIndex = i
                format = f
                break
            }
        }
        if (trackIndex < 0 || format == null) {
            extractor.release()
            return Result(emptyList(), 0, completed = false)
        }
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
        var sawInputEos = false
        var sawOutputEos = false
        var iterations = 0
        try {
            while (!sawOutputEos && picked.size < target && iterations < maxIterations) {
                iterations++
                if (!sawInputEos) {
                    val inIndex = codec.dequeueInputBuffer(2_000L)
                    if (inIndex >= 0) {
                        val buffer = codec.getInputBuffer(inIndex)!!
                        val size = extractor.readSampleData(buffer, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(inIndex, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawInputEos = true
                        } else {
                            codec.queueInputBuffer(inIndex, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                when (val outIndex = codec.dequeueOutputBuffer(info, 2_000L)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER, MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> Unit
                    else -> if (outIndex >= 0) {
                        val image: Image? = codec.getOutputImage(outIndex)
                        if (info.size > 0 && image != null) {
                            val ptsMs = info.presentationTimeUs / 1000L
                            while (nextTarget < target && targetsUs[nextTarget] / 1000L <= ptsMs) {
                                val bmp = imageToBitmap(image)
                                val out = if (resize) {
                                    Bitmap.createScaledBitmap(bmp, 256, 256, true)
                                        .also { if (it !== bmp) bmp.recycle() }
                                } else bmp
                                picked.add(Decoded(out, ptsMs))
                                nextTarget++
                            }
                            decoded++
                        }
                        codec.releaseOutputBuffer(outIndex, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) sawOutputEos = true
                    }
                }
            }
        } finally {
            runCatching { codec.stop() }
            runCatching { codec.release() }
            runCatching { extractor.release() }
        }
        val completed = sawOutputEos || picked.size >= target
        return Result(picked, decoded, completed)
    }

    fun imageToBitmap(image: Image): Bitmap {
        val planes = image.planes
        if (planes.size >= 3) return yuv420ToBitmap(image)
        val plane = planes[0]
        val rowPadding = plane.rowStride - plane.pixelStride * image.width
        val rawWidth = image.width + rowPadding / plane.pixelStride
        val raw = Bitmap.createBitmap(rawWidth, image.height, Bitmap.Config.ARGB_8888)
        plane.buffer.rewind()
        raw.copyPixelsFromBuffer(plane.buffer)
        return if (rawWidth != image.width) {
            Bitmap.createBitmap(raw, 0, 0, image.width, image.height).also { raw.recycle() }
        } else raw
    }

    private fun yuv420ToBitmap(image: Image): Bitmap {
        val w = image.width
        val h = image.height
        val yP = image.planes[0]; val uP = image.planes[1]; val vP = image.planes[2]
        val yB = yP.buffer; val uB = uP.buffer; val vB = vP.buffer
        val out = IntArray(w * h)
        for (row in 0 until h) {
            val yBase = row * yP.rowStride
            val uvRow = row / 2
            for (col in 0 until w) {
                val y = (yB.get(yBase + col * yP.pixelStride).toInt() and 0xFF) - 16
                val uIdx = uvRow * uP.rowStride + (col / 2) * uP.pixelStride
                val vIdx = uvRow * vP.rowStride + (col / 2) * vP.pixelStride
                val u = (uB.get(uIdx).toInt() and 0xFF) - 128
                val v = (vB.get(vIdx).toInt() and 0xFF) - 128
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
}