package org.echogesture.finalapp

import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import org.junit.Test

/**
 * Bisection probe: can a HandLandmarker even be created under instrumentation?
 *
 * Three separate test classes (ClipReplayAccuracyTest, GpuCpuParityTest, and a
 * previously-passing geometry test) now hang or crash with zero output on the
 * Vivo V2404. Common to all of them is creating a MediaPipe HandLandmarker.
 * This test does nothing else — no video, no decode, no classifier — so its
 * outcome separates "MediaPipe is broken under instrumentation on this device"
 * from "something in the replay logic is broken".
 *
 * It prints before and after each step, so a crash pinpoints the failing call
 * instead of vanishing silently.
 */
class MediaPipeProbeTest {

    @Test
    fun landmarkerCanBeCreatedAndClosed() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        println("PROBE step1 targetContext=${ctx.packageName}")

        for (delegate in listOf(Delegate.CPU, Delegate.GPU)) {
            println("PROBE step2 creating delegate=$delegate")
            val lm = try {
                HandLandmarker.createFromOptions(
                    ctx,
                    HandLandmarker.HandLandmarkerOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder()
                                .setModelAssetPath(ModelContract.DETECTOR_ASSET)
                                .setDelegate(delegate)
                                .build()
                        )
                        .setRunningMode(RunningMode.IMAGE)
                        .setNumHands(2)
                        .build()
                )
            } catch (t: Throwable) {
                println("PROBE create delegate=$delegate THREW ${t.javaClass.name}: ${t.message}")
                continue
            }
            println("PROBE step3 created delegate=$delegate")
            try {
                lm.close()
                println("PROBE step4 closed delegate=$delegate OK")
            } catch (t: Throwable) {
                println("PROBE close delegate=$delegate THREW ${t.javaClass.name}: ${t.message}")
            }
        }
        println("PROBE DONE")
    }

    /**
     * The classifier builds NINE TFLite interpreters, each with
     * `setNumThreads(4)` and XNNPACK on — up to 36 worker threads. Every test
     * that uses it blocks with 0% CPU before its first output line, so this
     * bisects whether the classifier alone reproduces that.
     *
     * Writes to a file because instrumentation stdout is discarded on this
     * device (a printing probe passes in 0.86 s and shows nothing).
     */
    @Test
    fun classifierCanBeBuilt() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        fun note(s: String) {
            println(s)
            runCatching {
                java.io.File(ctx.filesDir, "probe_classifier.txt").appendText(s + "\n")
            }
        }
        runCatching { java.io.File(ctx.filesDir, "probe_classifier.txt").writeText("") }
        note("CLS step1 loading ModelContract")
        val loaded = ModelContract.load(ctx.assets)
        note("CLS step2 loaded members=${loaded.members.size}")
        val t0 = System.currentTimeMillis()
        val c = LandmarkClassifier.create(loaded)
        note("CLS step3 created classifier in ${System.currentTimeMillis() - t0} ms")
        c.close()
        note("CLS step4 closed OK")
    }

    /**
     * Does DETECTING block, and is it delegate-specific?
     *
     * The parity test decodes fine then blocks at 0% CPU on its first
     * detection, so this narrows it to a single detect() call with no decode,
     * no classifier, and a synthetic Bitmap — separating "detect hangs" from
     * "GPU delegate hangs".
     */
    @Test
    fun detectWorksOnSyntheticBitmap() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        fun note(s: String) {
            println(s)
            runCatching {
                java.io.File(ctx.filesDir, "probe_detect.txt").appendText(s + "\n")
            }
        }
        runCatching { java.io.File(ctx.filesDir, "probe_detect.txt").writeText("") }

        val bmp = android.graphics.Bitmap.createBitmap(480, 640, android.graphics.Bitmap.Config.ARGB_8888)
        bmp.eraseColor(android.graphics.Color.rgb(40, 40, 40))
        note("DET synthetic bitmap ${bmp.width}x${bmp.height}")

        for (delegate in listOf(Delegate.CPU, Delegate.GPU)) {
            note("DET creating delegate=$delegate")
            val lm = try {
                HandLandmarker.createFromOptions(
                    ctx,
                    HandLandmarker.HandLandmarkerOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder()
                                .setModelAssetPath(ModelContract.DETECTOR_ASSET)
                                .setDelegate(delegate)
                                .build()
                        )
                        .setRunningMode(RunningMode.IMAGE)
                        .setNumHands(2)
                        .build()
                )
            } catch (t: Throwable) {
                note("DET create delegate=$delegate THREW ${t.javaClass.simpleName}")
                continue
            }
            note("DET created delegate=$delegate; calling detect()")
            try {
                val img = com.google.mediapipe.framework.image.BitmapImageBuilder(bmp).build()
                val r = lm.detect(img)
                img.close()
                note("DET detect delegate=$delegate OK hands=${r.landmarks().size}")
            } catch (t: Throwable) {
                note("DET detect delegate=$delegate THREW ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                runCatching { lm.close() }
            }
            note("DET delegate=$delegate closed; waiting before next")
            Thread.sleep(1500)
        }
        note("DET DONE")
    }
}