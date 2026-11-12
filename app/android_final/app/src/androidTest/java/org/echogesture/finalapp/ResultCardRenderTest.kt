package org.echogesture.finalapp

import android.view.View
import android.widget.ScrollView
import android.widget.TextView
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Renders the PRODUCTION result card from the real four-model ensemble.
 *
 * The emulator's synthetic camera never contains a hand, so the capture flow
 * correctly stops at NO_HANDS and RESULT cannot be reached by tapping through
 * the UI. Rather than fabricate a screenshot, this drives the real screen with
 * the real [LandmarkClassifier] over the pinned parity feature vector — the
 * same 24x255 input the Python reference produced for the real clip — and
 * asserts that what the card shows matches what the ensemble actually
 * predicted. Every number in the screenshot therefore came out of the four
 * shipped FP16 interpreters on this device. It is NOT a live-camera
 * prediction and must never be presented as one.
 *
 * The test then holds the rendered screen open until the harness signals that
 * it has taken a framebuffer capture, so the screenshot is a genuine grab of
 * the device screen rather than an off-screen re-render.
 */
@RunWith(AndroidJUnit4::class)
class ResultCardRenderTest {

    private val instrumentation
        get() = InstrumentationRegistry.getInstrumentation()

    private fun fixtureLines(): List<String> {
        val assets = instrumentation.context.assets
        return assets.open("parity_ensemble_runtime.txt").bufferedReader().readLines()
            .filter { it.isNotBlank() && !it.startsWith("#") }
    }

    private fun floats(line: String, n: Int): FloatArray {
        val v = line.trim().split(' ').filter { it.isNotEmpty() }
        assertEquals("float count", n, v.size)
        return FloatArray(n) { v[it].toFloat() }
    }

    private fun declared(field: String): java.lang.reflect.Field =
        CameraScreen::class.java.getDeclaredField(field).apply { isAccessible = true }

    /** True once the models are loaded AND the screen is idle in WAITING. */
    private fun settled(screen: CameraScreen): Boolean {
        if (!declared("modelReady").getBoolean(screen)) return false
        // bindUseCases() runs asynchronously after startup and publishes
        // WAITING when it finishes; a result rendered before that lands is
        // overwritten and the card disappears. Require a bound camera too.
        if (declared("camera").get(screen) == null) return false
        val state = declared("currentState").get(screen)
        return state?.toString() == "WAITING"
    }

    @Test
    fun resultCardShowsTheRealEnsembleOutput() {
        val assets = instrumentation.targetContext.assets
        val names = assets.list("")?.toList() ?: emptyList()
        val missing = ModelContract.MEMBER_SHA256.keys.filter { it !in names }
        org.junit.Assume.assumeTrue(
            "members not published ($missing absent); render test skipped", missing.isEmpty()
        )

        val l = fixtureLines()
        val classes = l[0].split(' ')[1].toInt()
        val features = floats(l[7 + 4 + 1].removePrefix("features "), 24 * 255)
        val expectedLabels = l[5 + 4].trim().split(' ')
        val expectedConfidence = l[6 + 4].trim().toFloat()

        // Real loader, real interpreters, real aggregation.
        val loaded = ModelContract.load(assets)
        val result = LandmarkClassifier.create(loaded).use { it.predict(features) }

        assertEquals("top-1 label", expectedLabels[0], result.prediction)
        assertEquals("confidence", expectedConfidence, result.confidence, 1e-3f)
        assertEquals(classes, result.averagedLogits!!.size)

        val activity = ActivityLaunch.fresh()

        try {
            val cameraField = MainActivity::class.java.getDeclaredField("cameraScreen")
                .apply { isAccessible = true }
            val screen = cameraField.get(activity) as CameraScreen

            val deadline = System.currentTimeMillis() + 40_000
            // Wait for the screen to actually settle, not merely for the
            // models to load: onModelLoaded() and bindUseCases() each publish
            // WAITING, and a result rendered in that gap is immediately
            // overwritten and the card is hidden again. Require the state to
            // hold steady across several samples.
            var stable = 0
            while (stable < 6) {
                if (System.currentTimeMillis() > deadline) {
                    throw AssertionError("MainActivity never settled into a waiting state")
                }
                stable = if (settled(screen)) stable + 1 else 0
                Thread.sleep(250)
            }
            instrumentation.waitForIdleSync()

            // Same private method the capture path uses to publish a result.
            val render = CameraScreen::class.java.getDeclaredMethod(
                "renderResult",
                LandmarkClassifier.Result::class.java,
                java.lang.Long.TYPE
            ).apply { isAccessible = true }

            instrumentation.runOnMainSync {
                render.invoke(screen, result, result.inferenceMillis)
                val card = activity.findViewById<View>(R.id.resultCard)
                // Settle the 180 ms reveal so the capture cannot catch a
                // half-transparent card.
                card.animate().cancel()
                card.alpha = 1f
                card.translationY = 0f
            }
            instrumentation.waitForIdleSync()

            // The card sits below the fold; bring it into the viewport.
            val scroll = screen.root as ScrollView
            instrumentation.runOnMainSync {
                val card = activity.findViewById<View>(R.id.resultCard)
                var y = 0
                var v: View? = card
                while (v != null && v !== scroll) {
                    y += v.top
                    v = v.parent as? View
                }
                scroll.scrollTo(0, y - 24)
            }
            instrumentation.waitForIdleSync()
            Thread.sleep(400)

            val card = activity.findViewById<View>(R.id.resultCard)
            val caption = activity.findViewById<TextView>(R.id.captionText)
            val badge = activity.findViewById<TextView>(R.id.confidenceBadge)
            val alternatives = activity.findViewById<TextView>(R.id.alternativesText)
            val latency = activity.findViewById<TextView>(R.id.latencyText)

            assertEquals("card must be visible", View.VISIBLE, card.visibility)
            assertEquals("caption is the model's own label", result.prediction, caption.text.toString())
            assertTrue(
                "badge shows the real score, was '${badge.text}'",
                badge.text.toString().contains("%.1f".format(result.confidence * 100f))
            )
            assertTrue(
                "alternatives list the runner-ups, was '${alternatives.text}'",
                alternatives.text.toString().contains(result.top5[1])
            )
            assertTrue(
                "latency is a measured non-zero duration, was '${latency.text}'",
                result.inferenceMillis > 0 &&
                    latency.text.toString().contains(result.inferenceMillis.toString())
            )

            // Hold the screen while the harness grabs the framebuffer. The ready
            // marker is a file, not a log line: a stale log line from an
            // earlier run once raced the capture and photographed the camera
            // screen instead of the result. It lives in the app's own external
            // directory because scoped storage denies writes to /sdcard root.
            val dir: File = instrumentation.targetContext.getExternalFilesDir(null)!!
            val ready = File(dir, "eg_result_ready")
            val go = File(dir, "eg_result_shot_done")
            ready.writeText("${result.prediction} ${result.confidence}\n")
            android.util.Log.i("ResultCardRenderTest", "ready at ${ready.absolutePath}")
            val shotDeadline = System.currentTimeMillis() + 90_000
            while (!go.exists() && System.currentTimeMillis() < shotDeadline) Thread.sleep(250)
            ready.delete()
            android.util.Log.i("ResultCardRenderTest", "hold released")
        } finally {
            instrumentation.runOnMainSync { activity.finish() }
        }
    }
}