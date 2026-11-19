package org.echogesture.finalapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import java.nio.ByteBuffer

/**
 * Sequential video decoding, replacing per-frame seeking.
 *
 * Why this exists. `ClipAnalyzer` used `MediaMetadataRetriever.getFrameAtTime`,
 * which seeks and decodes from a keyframe for EVERY requested frame. Measured on
 * the Vivo V2404: 55 frames cost 19,872 ms of decoding — 87% of a 22.7 s run,
 * against 2,279 ms of actual hand detection. Switching to a scaled seek got it
 * to 8,698 ms; it is still the dominant cost, and it grows with the number of
 * frames requested, so a 2-minute clip is hopeless.
 *
 * A decoder is the right instrument: `MediaExtractor` reads one compressed
 * stream end to end, `MediaCodec` decodes it once, and frames are taken as they
 * come out. One pass over the file instead of one seek per frame.
 *
 * Output format. MediaCodec produces YUV_420_888 through an [ImageReader]; the
 * classifier needs ARGB. The conversion is done here, downscaled, once per
 * accepted frame — the same "decode at display size, never allocate the full
 * bitmap" principle that made the seeking version 2.3x faster.
 *
 * Frame selection. The decoder emits every frame; only [targetFps] worth are
 * kept, chosen by presentation time so the sampling matches what
 * `read_frames()` does on the training path (evenly spaced across the clip).
 *
 * Not used when the clip is unreadable or the codec refuses the format —
 * [ClipAnalyzer] falls back to the retriever path, which is slower but works on
 * every file the retriever accepts.
 */
object SequentialVideoDecoder {

    private const val TAG = "SeqVideoDecoder"

    class Result(
        val bitmaps: List<Bitmap>,
        val ptsMs: List<Long>,
        val codecName: String,
        val decodeMs: Long
    )

    /**
     * Decodes [uri] and returns frames at approximately [targetFps], scaled so
     * the longest edge is at most [maxEdge].
     *
     * Returns null when the codec cannot handle the file, so the caller can
     * fall back rather than fail.
     */
    fun decode(
        context: Context,
        uri: Uri,
        targetFps: Int,
        maxEdge: Int,
        maxFrames: Int
    ): Result? {
        val started = SystemClock.elapsedRealtime()
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        var reader: ImageReader? = null
        var thread: HandlerThread? = null
        try {
            context.contentResolver.openFileDescriptor(uri, "r")?.use {
                extractor.setDataSource(it.fileDescriptor)
            } ?: return null

            var trackIndex = -1
            var format: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                if (f.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) {
                    trackIndex = i; format = f; break
                }
            }
            if (trackIndex < 0 || format == null) return null
            extractor.selectTrack(trackIndex)
            val mime = format.getString(MediaFormat.KEY_MIME)!!
            val srcW = format.getInteger(MediaFormat.KEY_WIDTH)
            val srcH = format.getInteger(MediaFormat.KEY_HEIGHT)
            val durationUs = format.getLong(MediaFormat.KEY_DURATION)

            // Scale factors, applied when converting YUV to ARGB.
            val longEdge = maxOf(srcW, srcH)
            val scale = if (longEdge > maxEdge) maxEdge.toFloat() / longEdge else 1f
            val outW = (srcW * scale).toInt().coerceAtLeast(2) and 0xFFFFFFFE.toInt()
            val outH = (srcH * scale).toInt().coerceAtLeast(2) and 0xFFFFFFFE.toInt()

            codec = MediaCodec.createDecoderByType(mime)

            val bitmaps = ArrayList<Bitmap>()
            val pts = ArrayList<Long>()
            val stepUs = 1_000_000L / targetFps
            var nextTargetUs = 0L
            var lastTakenUs = -1L

            thread = HandlerThread("seq-decoder").also { it.start() }
            reader = ImageReader.newInstance(
                srcW, srcH, ImageFormat.YUV_420_888, 4
            )

            // Frames are drained on the reader's own thread as the codec renders
            // into its surface. Configure() MUST receive that surface and every
            // output buffer MUST be released with render=true, or the listener
            // never fires at all.
            reader.setOnImageAvailableListener({ r ->
                val image = try { r.acquireLatestImage() } catch (t: Throwable) { null }
                    ?: return@setOnImageAvailableListener
                try {
                    val tUs = image.timestamp / 1000L
                    if (tUs >= nextTargetUs && bitmaps.size < maxFrames) {
                        val bmp = yuvToBitmap(image, outW, outH)
                        if (bmp != null) {
                            synchronized(bitmaps) {
                                if (bitmaps.size < maxFrames) {
                                    bitmaps.add(bmp); pts.add(tUs); lastTakenUs = tUs
                                } else {
                                    bmp.recycle()
                                }
                            }
                            nextTargetUs = tUs + stepUs
                        }
                    }
                } catch (t: Throwable) {
                    Log.w(TAG, "frame drop: ${t.javaClass.simpleName}")
                } finally {
                    image.close()
                }
            }, Handler(thread.looper))

            codec.configure(format, reader.surface, null, 0)
            codec.start()

            val bufferInfo = MediaCodec.BufferInfo()
            var sawInputEos = false
            var sawOutputEos = false
            while (!sawOutputEos) {
                if (!sawInputEos) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(
                                inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            sawInputEos = true
                        } else {
                            codec.queueInputBuffer(
                                inIdx, 0, size, extractor.sampleTime,
                                if (extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0)
                                    MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
                            )
                            extractor.advance()
                        }
                    }
                }
                val outIdx = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                when {
                    outIdx >= 0 -> {
                        val isEos = bufferInfo.flags and
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        // render=true: hand the frame to the ImageReader surface.
                        codec.releaseOutputBuffer(outIdx, true)
                        if (isEos) sawOutputEos = true
                    }
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val f = codec.outputFormat
                        Log.i(TAG, "output format ${f.getInteger(MediaFormat.KEY_WIDTH)}x" +
                            "${f.getInteger(MediaFormat.KEY_HEIGHT)}")
                    }
                    outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                }
                if (bitmaps.size >= maxFrames) break
                if (durationUs > 0 && bufferInfo.presentationTimeUs > durationUs) break
            }

            // Let the reader drain any queued frame before tearing the surface
            // down; releasing it while an image is in flight aborts the process.
            SystemClock.sleep(60)

            val elapsed = SystemClock.elapsedRealtime() - started
            return Result(bitmaps, pts, codec.name, elapsed)
        } catch (t: Throwable) {
            Log.w(TAG, "sequential decode unavailable: ${t.javaClass.simpleName}: ${t.message}")
            return null
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            runCatching { reader?.close() }
            runCatching { thread?.quitSafely() }
            runCatching { extractor.release() }
        }
    }

    /**
     * YUV_420_888 -> ARGB_8888, downscaling while converting.
     *
     * Nearest-neighbour on a sampled grid rather than a full-resolution convert
     * followed by a resize: the full-size ARGB bitmap would be width*height*4
     * bytes for a frame whose pixels are about to be thrown away.
     */
    private fun yuvToBitmap(image: Image, outW: Int, outH: Int): Bitmap? {
        val y = image.planes[0]
        val u = image.planes[1]
        val v = image.planes[2]
        val srcW = image.width
        val srcH = image.height
        val yBuf = y.buffer
        val uBuf = u.buffer
        val vBuf = v.buffer
        val yRowStride = y.rowStride
        val yPixStride = y.pixelStride
        val uRowStride = u.rowStride
        val uPixStride = u.pixelStride
        val vRowStride = v.rowStride
        val vPixStride = v.pixelStride

        val pixels = IntArray(outW * outH)
        val yArr = ByteArray(yBuf.remaining()).also { yBuf.get(it) }
        val uArr = ByteArray(uBuf.remaining()).also { uBuf.get(it) }
        val vArr = ByteArray(vBuf.remaining()).also { vBuf.get(it) }

        for (oy in 0 until outH) {
            val sy = (oy.toLong() * srcH / outH).toInt()
            val yRow = sy * yRowStride
            val uvRow = (sy / 2)
            for (ox in 0 until outW) {
                val sx = (ox.toLong() * srcW / outW).toInt()
                val yi = yRow + sx * yPixStride
                if (yi >= yArr.size) continue
                val yv = (yArr[yi].toInt() and 0xFF) - 16
                val ui = uvRow * uRowStride + (sx / 2) * uPixStride
                val vi = uvRow * vRowStride + (sx / 2) * vPixStride
                val uv = if (ui < uArr.size && vi < vArr.size) {
                    (uArr[ui].toInt() and 0xFF) - 128 to (vArr[vi].toInt() and 0xFF) - 128
                } else 0 to 0
                val yy = (if (yv < 0) 0 else yv) * 1.164f
                val r = (yy + 1.596f * uv.second).toInt().coerceIn(0, 255)
                val g = (yy - 0.813f * uv.second - 0.391f * uv.first).toInt().coerceIn(0, 255)
                val b = (yy + 2.018f * uv.first).toInt().coerceIn(0, 255)
                pixels[oy * outW + ox] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(pixels, outW, outH, Bitmap.Config.ARGB_8888)
    }
}
