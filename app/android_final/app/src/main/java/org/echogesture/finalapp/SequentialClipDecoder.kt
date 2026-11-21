package org.echogesture.finalapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo.CodecCapabilities
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.SystemClock

/** Reads short upright clips in one pass. The retriever handles other formats and providers. */
internal object SequentialClipDecoder {
    private const val MAX_DURATION_MS = 20_000L
    private const val MAX_DECODE_MS = 12_000L
    private const val MAX_STALL_MS = 2_000L
    private const val POLL_US = 2_000L

    fun sample(
        context: Context,
        uri: Uri,
        durationMs: Long,
        samples: Int,
        maxEdge: Int,
        keepGoing: () -> Boolean,
        accept: (Bitmap, Long) -> Unit,
    ): Int {
        if (durationMs > MAX_DURATION_MS) return 0
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            val track = (0 until extractor.trackCount).firstOrNull {
                extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
            } ?: return 0
            val format = extractor.getTrackFormat(track)
            if (format.containsKey(MediaFormat.KEY_ROTATION) && format.getInteger(MediaFormat.KEY_ROTATION) != 0) {
                return 0
            }
            if (maxOf(format.getInteger(MediaFormat.KEY_WIDTH), format.getInteger(MediaFormat.KEY_HEIGHT)) > 1920) {
                return 0
            }
            extractor.selectTrack(track)
            val mime = checkNotNull(format.getString(MediaFormat.KEY_MIME))
            val decoder = MediaCodec.createDecoderByType(mime)
            codec = decoder
            val colorFormats = decoder.codecInfo.getCapabilitiesForType(mime).colorFormats
            val preferred = listOf(
                CodecCapabilities.COLOR_FormatYUV420Flexible,
                CodecCapabilities.COLOR_FormatYUV420Planar,
                CodecCapabilities.COLOR_FormatYUV420SemiPlanar
            ).firstOrNull { it in colorFormats } ?: return 0
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, preferred)
            decoder.configure(format, null, null, 0)
            decoder.start()

            val started = SystemClock.elapsedRealtime()
            var progress = started
            var inputFinished = false
            var outputFinished = false
            var count = 0
            var firstPts: Long? = null
            val stepUs = durationMs * 1000 / samples
            var nextUs = 0L
            val info = MediaCodec.BufferInfo()
            while (!outputFinished && count < samples && keepGoing()) {
                val now = SystemClock.elapsedRealtime()
                if (now - started > MAX_DECODE_MS || now - progress > MAX_STALL_MS) return 0
                if (!inputFinished) {
                    val index = decoder.dequeueInputBuffer(POLL_US)
                    if (index >= 0) {
                        val size = extractor.readSampleData(checkNotNull(decoder.getInputBuffer(index)), 0)
                        inputFinished = size < 0
                        if (inputFinished) {
                            decoder.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        } else {
                            decoder.queueInputBuffer(index, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                        progress = SystemClock.elapsedRealtime()
                    }
                }
                val output = decoder.dequeueOutputBuffer(info, POLL_US)
                if (output < 0) continue
                progress = SystemClock.elapsedRealtime()
                outputFinished = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                var bitmap: Bitmap? = null
                var pts = 0L
                try {
                    if (info.size > 0) {
                        val origin = firstPts ?: info.presentationTimeUs.also { firstPts = it }
                        pts = info.presentationTimeUs - origin
                        if (pts >= nextUs) {
                            bitmap = decoder.getOutputImage(output)?.use { toBitmap(it, maxEdge) }
                                ?: return 0
                        }
                    }
                } finally {
                    decoder.releaseOutputBuffer(output, false)
                }
                if (bitmap != null) {
                    accept(bitmap, pts / 1000)
                    count++
                    nextUs += stepUs
                    while (nextUs <= pts) nextUs += stepUs
                }
            }
            return count
        } catch (_: RuntimeException) {
            return 0
        } catch (_: java.io.IOException) {
            return 0
        } finally {
            runCatching { codec?.release() }
            extractor.release()
        }
    }

    private fun toBitmap(image: Image, maxEdge: Int): Bitmap? {
        if (image.format != ImageFormat.YUV_420_888 || image.planes.size != 3) return null
        val crop = image.cropRect
        val scale = minOf(1f, maxEdge.toFloat() / maxOf(crop.width(), crop.height()))
        val width = (crop.width() * scale).toInt().coerceAtLeast(1)
        val height = (crop.height() * scale).toInt().coerceAtLeast(1)
        val (y, u, v) = image.planes
        val planes = image.planes.map { plane ->
            val buffer = plane.buffer
            ByteArray(buffer.remaining()).also { buffer.get(it) }
        }
        val pixels = IntArray(width * height)
        for (row in 0 until height) {
            val sourceY = crop.top + row * crop.height() / height
            val yRow = sourceY * y.rowStride
            val uRow = sourceY / 2 * u.rowStride
            val vRow = sourceY / 2 * v.rowStride
            for (column in 0 until width) {
                val sourceX = crop.left + column * crop.width() / width
                val yByte = planes[0].getOrNull(yRow + sourceX * y.pixelStride) ?: return null
                val uByte = planes[1].getOrNull(uRow + sourceX / 2 * u.pixelStride) ?: return null
                val vByte = planes[2].getOrNull(vRow + sourceX / 2 * v.pixelStride) ?: return null
                val luminance = ((yByte.toInt() and 255) - 16).coerceAtLeast(0) * 298
                val cb = (uByte.toInt() and 255) - 128
                val cr = (vByte.toInt() and 255) - 128
                val red = ((luminance + 409 * cr + 128) shr 8).coerceIn(0, 255)
                val green = ((luminance - 100 * cb - 208 * cr + 128) shr 8).coerceIn(0, 255)
                val blue = ((luminance + 516 * cb + 128) shr 8).coerceIn(0, 255)
                pixels[row * width + column] = (255 shl 24) or (red shl 16) or (green shl 8) or blue
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }
}
