package org.echogesture.finalapp

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo.CodecCapabilities
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.SystemClock
import android.util.Log

/**
 * Sequential video decoding, replacing per-frame seeking.
 *
 * WHY. `ClipAnalyzer` used `MediaMetadataRetriever.getFrameAtTime`, which seeks
 * to a keyframe and decodes forward for EVERY requested frame. Measured on the
 * Vivo V2404: 55 frames cost 8.7-21 s of decoding (the spread is thermal
 * throttling), against 2.3 s of actual hand detection. The cost grows with the
 * number of frames requested, so a long clip is hopeless.
 *
 * `MediaExtractor` reads one compressed stream end to end and `MediaCodec`
 * decodes it once; frames are taken as they come out. One pass instead of one
 * seek per frame.
 *
 * WHY BUFFER MODE, NOT A SURFACE. The first attempt at this used an
 * `ImageReader` surface and killed the process:
 *
 *     JNI DETECTED ERROR: non-zero capacity for nullptr pointer: 1
 *       in call to NewDirectByteBuffer
 *       from ImageReader$SurfaceImage.nativeCreatePlanes
 *
 * That is a native abort with no Java stack -- the app vanishes rather than
 * throwing. The reader had been created from the SOURCE format while the codec
 * produced its own output format, so the surface had no valid plane buffers.
 *
 * Buffer mode (`configure(format, null, null, 0)` then `getOutputImage(index)`)
 * avoids that whole class of failure: there is no surface to size wrongly and no
 * native plane allocation to get wrong. This exact path is already proven on
 * this device -- `DirectClipParityTest` decodes the same way and both of its
 * tests pass, including the one that compares against the shipped pipeline.
 *
 * OUTPUT. Codec output is YUV_420_888; the classifier needs ARGB. Conversion is
 * done once per accepted frame, downscaling while converting so the full-size
 * bitmap is never allocated.
 *
 * Frames are chosen by presentation time at roughly [targetFps], matching what
 * `read_frames()` does on the training path.
 */
object SequentialVideoDecoder {

    private const val TAG = "SeqVideoDecoder"

    class Result(
        val bitmaps: List<Bitmap>,
        val ptsMs: List<Long>,
        val codecName: String,
        val decodeMs: Long,
        /** Stage counters, so a zero-frame result can be diagnosed without logcat. */
        val diag: String = ""
    )

    /**
     * Decodes [uri] and returns frames at approximately [targetFps], scaled so
     * the longest edge is at most [maxEdge].
     *
     * Returns null when the codec cannot handle the file, so the caller can fall
     * back to the retriever rather than fail.
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

            codec = MediaCodec.createDecoderByType(mime)
            // Buffer mode with an EXPLICIT colour format.
            //
            // getOutputImage() only returns an Image when the codec is emitting
            // a READABLE format. Left to itself, c2.qti.avc.decoder chose an
            // opaque one and getOutputImage() returned null on every frame --
            // the decoder ran cleanly (480 ms, no crash) and produced nothing.
            // Asking for YUV420 flexible (or planar as a fallback) makes the
            // output readable.
            val chosen = pickColorFormat(codec, mime)
            if (chosen >= 0) format.setInteger(MediaFormat.KEY_COLOR_FORMAT, chosen)
            codec.configure(format, null, null, 0)
            codec.start()

            val bitmaps = ArrayList<Bitmap>()
            val pts = ArrayList<Long>()
            val stepUs = 1_000_000L / targetFps
            var nextTargetUs = 0L
            var tooManyFrames = false

            val bufferInfo = MediaCodec.BufferInfo()
            var sawInputEos = false
            var sawOutputEos = false
            // Hard iteration cap: a codec that produces no usable images must
            // fail rather than spin. The parity tests hung for over ten minutes
            // on exactly this before they were bounded.
            var spins = 0
            val maxSpins = 400_000
            // Diagnostics: distinguishes "codec produced no buffers" from
            // "it produced buffers we could not read", which have different
            // fixes and cannot be told apart from a frame count of zero.
            var outBuffers = 0
            var withSize = 0
            var imageNull = 0
            var imageOk = 0
            var fmtChanges = 0
            var outW = 0
            var outH = 0

            while (!sawOutputEos && bitmaps.size < maxFrames) {
                if (++spins > maxSpins) break

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

                when (val outIdx = codec.dequeueOutputBuffer(bufferInfo, 10_000)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        fmtChanges++
                        val f = codec.outputFormat
                        outW = f.getInteger(MediaFormat.KEY_WIDTH)
                        outH = f.getInteger(MediaFormat.KEY_HEIGHT)
                        Log.i(TAG, "output format changed: ${outW}x${outH} " +
                            "colour=${runCatching { f.getInteger(MediaFormat.KEY_COLOR_FORMAT) }.getOrDefault(-1)}")
                    }
                    else -> if (outIdx >= 0) {
                        outBuffers++
                        val isEos = bufferInfo.flags and
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        if (bufferInfo.size > 0) withSize++
                        if (true) {
                            val image: Image? = try {
                                codec.getOutputImage(outIdx)
                            } catch (t: Throwable) {
                                null
                            }
                            if (image == null) imageNull++ else imageOk++
                            if (image != null) {
                                try {
                                    val tUs = bufferInfo.presentationTimeUs
                                    if (tUs >= nextTargetUs) {
                                        val bmp = yuvToBitmap(
                                            image,
                                            maxEdge,
                                            maxFrames,
                                            bitmaps.size
                                        )
                                        if (bmp != null) {
                                            if (bitmaps.size < maxFrames) {
                                                bitmaps.add(bmp)
                                                pts.add(tUs / 1000L)
                                            } else {
                                                bmp.recycle()
                                            }
                                            nextTargetUs = tUs + stepUs
                                        }
                                    }
                                } finally {
                                    image.close()
                                }
                            }
                        }
                        codec.releaseOutputBuffer(outIdx, false)
                        if (isEos) sawOutputEos = true
                        if (bitmaps.size >= maxFrames) tooManyFrames = true
                    }
                }
            }

            val elapsed = SystemClock.elapsedRealtime() - started
            Log.i(
                TAG,
                "diag outBuffers=$outBuffers withSize=$withSize imageNull=$imageNull " +
                    "imageOk=$imageOk fmtChanges=$fmtChanges fmt=${outW}x${outH} " +
                    "spins=$spins"
            )
            Log.i(
                TAG,
                "sequential decode ${bitmaps.size} frames in ${elapsed}ms " +
                    "codec=${codec.name} cap=${if (tooManyFrames) "hit" else "clear"}"
            )
            return Result(
                bitmaps, pts, codec.name, elapsed,
                "outBuffers=$outBuffers withSize=$withSize imageNull=$imageNull " +
                    "imageOk=$imageOk fmtChanges=$fmtChanges fmt=${outW}x${outH} spins=$spins"
            )
        } catch (t: Throwable) {
            Log.w(TAG, "sequential decode unavailable: ${t.javaClass.simpleName}: ${t.message}")
            return null
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            runCatching { extractor.release() }
        }
    }

    /**
     * The colour format to request, or -1 if the codec offers none readable.
     *
     * YUV420_FLEXIBLE first: it maps to YUV_420_888 and MediaCodec advertises it
     * for both YUV420Planar and YUV420PackedPlanar, so getOutputImage() can
     * build an Image with real planes. Planar is the fallback for older
     * decoders. Anything else (surface-only formats) is unreadable here.
     */
    private fun pickColorFormat(codec: MediaCodec, mime: String): Int {
        return runCatching {
            val caps = codec.codecInfo.getCapabilitiesForType(mime)
            val supported = caps.colorFormats.toSet()
            when {
                CodecCapabilities.COLOR_FormatYUV420Flexible in supported ->
                    CodecCapabilities.COLOR_FormatYUV420Flexible
                CodecCapabilities.COLOR_FormatYUV420Planar in supported ->
                    CodecCapabilities.COLOR_FormatYUV420Planar
                CodecCapabilities.COLOR_FormatYUV420SemiPlanar in supported ->
                    CodecCapabilities.COLOR_FormatYUV420SemiPlanar
                else -> -1
            }
        }.getOrDefault(-1)
    }

    /**
     * YUV_420_888 -> ARGB_8888, downscaling while converting.
     *
     * Nearest-neighbour on a sampled grid rather than converting at full size
     * and resizing: the full-size ARGB bitmap would be width*height*4 bytes for
     * pixels that are about to be discarded.
     */
    private fun yuvToBitmap(
        image: Image,
        maxEdge: Int,
        maxFrames: Int,
        collected: Int
    ): Bitmap? {
        val srcW = image.width
        val srcH = image.height
        if (srcW <= 0 || srcH <= 0 || collected >= maxFrames) return null

        // Fit the longest edge; uniform so aspect is preserved.
        val longEdge = maxOf(srcW, srcH)
        val scale = if (longEdge > maxEdge) maxEdge.toFloat() / longEdge else 1f
        var outW = (srcW * scale).toInt().coerceAtLeast(2)
        var outH = (srcH * scale).toInt().coerceAtLeast(2)

        val planes = image.planes
        if (planes.size < 3) return null
        val y = planes[0]; val u = planes[1]; val v = planes[2]
        val yBuf = y.buffer; val uBuf = u.buffer; val vBuf = v.buffer
        val yArr = ByteArray(yBuf.remaining()).also { yBuf.get(it) }
        val uArr = ByteArray(uBuf.remaining()).also { uBuf.get(it) }
        val vArr = ByteArray(vBuf.remaining()).also { vBuf.get(it) }
        val yRowStride = y.rowStride; val yPixStride = y.pixelStride
        val uRowStride = u.rowStride; val uPixStride = u.pixelStride
        val vRowStride = v.rowStride; val vPixStride = v.pixelStride

        val pixels = IntArray(outW * outH)
        for (oy in 0 until outH) {
            val sy = (oy.toLong() * srcH / outH).toInt()
            val yRow = sy * yRowStride
            val uvRow = sy / 2
            for (ox in 0 until outW) {
                val sx = (ox.toLong() * srcW / outW).toInt()
                val yi = yRow + sx * yPixStride
                if (yi < 0 || yi >= yArr.size) continue
                val yv = (yArr[yi].toInt() and 0xFF) - 16
                val ui = uvRow * uRowStride + (sx / 2) * uPixStride
                val vi = uvRow * vRowStride + (sx / 2) * vPixStride
                val uu = if (ui in uArr.indices) (uArr[ui].toInt() and 0xFF) - 128 else 0
                val vv = if (vi in vArr.indices) (vArr[vi].toInt() and 0xFF) - 128 else 0
                val yy = (if (yv < 0) 0 else yv) * 1.164f
                val r = (yy + 1.596f * vv).toInt().coerceIn(0, 255)
                val g = (yy - 0.813f * vv - 0.391f * uu).toInt().coerceIn(0, 255)
                val b = (yy + 2.018f * uu).toInt().coerceIn(0, 255)
                pixels[oy * outW + ox] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return runCatching {
            Bitmap.createBitmap(pixels, outW, outH, Bitmap.Config.ARGB_8888)
        }.getOrNull()
    }
}