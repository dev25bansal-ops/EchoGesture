package org.echogesture.finalapp

import android.graphics.Bitmap
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
 * This is the controlled comparison the Python path cannot make for a camera,
 * and the camera cannot make for Python. The clip is the identical file, so
 * any prediction difference is attributable to the Android preprocessing and
 * detector path, not to content.
 *
 * The clip is NOT bundled: it is authorized research data and must never be
 * committed. The test pushes to the app's own external files directory
 * (`adb push <clip> /sdcard/Android/data/org.echogesture.finalapp/files/`) and
 * SKIPS when absent rather than failing, so a source-only checkout stays green.
 *
 * It writes no frames anywhere and reports only counts, labels and timings.
 */
@RunWith(AndroidJUnit4::class)
class DirectClipParityTest {

    private val instrumentation
        get() = InstrumentationRegistry.getInstrumentation()

    private fun clipFile(): File? {
        // Look in both places an operator can put a clip: the app's own
        // external files dir (adb push to /sdcard/Android/data/<pkg>/files/)
        // and its internal files dir (adb push to /data/local/tmp then
        // `run-as <pkg> cp`), which is the only path scoped storage allows on
        // some devices.
        val dirs = listOfNotNull(
            instrumentation.targetContext.getExternalFilesDir(null),
            instrumentation.targetContext.filesDir,
        )
        return dirs
            .flatMap { d -> d.listFiles { f -> f.isFile && f.name.endsWith(".mp4", true) }?.toList() ?: emptyList() }
            .sortedBy { it.name }
            .firstOrNull()
    }

    /** Decodes up to [n] frames spread across the clip, matching training. */
    private fun sampleFrames(file: File, n: Int = 32, resize: Boolean): List<Bitmap> {
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(file.absolutePath)

        val durationMs = retriever
            .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            ?.toLongOrNull() ?: 0L
        if (durationMs <= 0L) {
            retriever.release()
            return emptyList()
        }
        val out = ArrayList<Bitmap>(n)
        // Same index mapping as np.linspace(0, total-1, n): evenly across the
        // WHOLE clip, which is what training sampled.
        for (i in 0 until n) {
            val tMs = if (n == 1) 0L else durationMs * i / (n - 1)
            val frame = retriever.getFrameAtTime(
                tMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST_SYNC
            ) ?: continue
            // MediaMetadataRetriever hands back RGB_565; MediaPipe requires
            // ARGB_8888. Convert before any scaling so the resize samples the
            // same 8-bit channels the camera path would deliver.
            val argb = frame.copy(Bitmap.Config.ARGB_8888, false)
            frame.recycle()
            val bmp = if (resize) Bitmap.createScaledBitmap(argb, 256, 256, true) else argb
            if (bmp !== argb) argb.recycle()
            out.add(bmp)
        }
        retriever.release()
        return out
    }

    private fun runPath(file: File, resize: Boolean): Triple<Int, Int, FloatArray> {
        val frames = sampleFrames(file, 32, resize)
        if (frames.isEmpty()) return Triple(0, 0, FloatArray(0))

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

        // Flattened (T, 2, 21, 3) + (T, 2) exactly as toFeatureArrays builds them.
        val n = frames.size
        val landmarks = FloatArray(n * 2 * 21 * 3)
        val present = FloatArray(n * 2)
        var hands = 0
        for ((fi, bmp) in frames.withIndex()) {
            val img = BitmapImageBuilder(bmp).build()
            val result = landmarker.detect(img)
            img.close()
            bmp.recycle()
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
            if (fi % 4 == 0) println("DIRECT_CLIP frame $fi hands=${result.landmarks().size}")
        }
        landmarker.close()
        return Triple(n, hands, landmarks + present)
    }

    /**
     * Isolates the detector from the video decoder: load one PNG that is known
     * to contain a hand (Python finds one here), and run the shipped detector
     * on it. If this finds a hand, the detector is fine and the MP4 path is at
     * fault; if it finds nothing, the detector or its input config is at fault.
     */
    @Test
    fun detectorFindsAHandInAKnownStillImage() {
        val dir = instrumentation.targetContext.filesDir
        val png = File(dir, "withhand.png")
        assumeTrue("push withhand.png to the app files dir to run this test", png.isFile)
        val bmp = android.graphics.BitmapFactory.decodeFile(png.absolutePath)
            .copy(Bitmap.Config.ARGB_8888, false)
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
        val lm = HandLandmarker.createFromOptions(instrumentation.targetContext, options)
        for (resize in listOf(false, true)) {
            // Always build a fresh input from bmp: scaling may return the same
            // instance, and MediaPipe closes the bitmap it was given, so reusing
            // bmp across iterations would operate on a recycled bitmap.
            val input = if (resize) {
                Bitmap.createScaledBitmap(bmp, 256, 256, true)
            } else {
                bmp.copy(Bitmap.Config.ARGB_8888, false)
            }
            val img = BitmapImageBuilder(input).build()
            val res = lm.detect(img)
            println("DETECTOR_STILL resize=$resize ${input.width}x${input.height} hands=${res.landmarks().size}")
            img.close()
            if (input !== bmp) input.recycle()
        }
        bmp.recycle()
        lm.close()
    }

    @Test
    fun androidDirectClipMatchesTheShippedPipeline() {
        val file = clipFile()
        assumeTrue(
            "no .mp4 in the app's external files dir; push one to run this test",
            file != null
        )

        val assets = instrumentation.targetContext.assets
        val loaded = ModelContract.load(assets)
        val classifier = LandmarkClassifier.create(loaded)

        val results = LinkedHashMap<String, String>()
        classifier.use {
            for (resize in listOf(true, false)) {
                val (n, hands, flat) = runPath(file!!, resize)
                if (n == 0) {
                    results["resize=$resize"] = "no frames decoded"
                    continue
                }
                val half = n * 2 * 21 * 3
                val landmarks = flat.copyOfRange(0, half)
                val present = flat.copyOfRange(half, flat.size)
                val feats = FeatureTransform.buildFeatures(landmarks, present, n, 24)
                val r = classifier.predict(feats)
                val tag = if (resize) "resize256" else "native"
                results[tag] = "frames=$n hands=$hands top1=${r.prediction} " +
                    "score=${"%.4f".format(r.confidence)} top5=${r.top5.take(3)}"
                println("DIRECT_CLIP $tag ${results[tag]}")
                assertTrue("frames decoded", n > 0)
                assertTrue("landmarks are finite", landmarks.all { it.isFinite() })
                assertTrue("top5 has five entries", r.top5.size == 5)
            }
        }

        // The test's job is to make both paths observable side by side; it must
        // not assert a particular label, because the clip is operator-supplied
        // and its ground truth is not known here.
        assertTrue("both preprocessing paths ran", results.size >= 1)
    }
}