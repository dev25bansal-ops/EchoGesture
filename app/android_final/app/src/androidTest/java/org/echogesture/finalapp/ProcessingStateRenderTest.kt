package org.echogesture.finalapp

import android.content.Intent
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Renders the PRODUCTION PROCESSING state so it can be photographed.
 *
 * The emulator's synthetic camera never contains a hand, and inference on the
 * real ensemble finishes in roughly 70 ms, so the state is written to the UI
 * thread and cleared again well inside one screencap. This invokes the very
 * same private `render()` the capture path calls at its PROCESSING step, then
 * asserts the state-specific affordances are what a user would actually see:
 * the button is disabled and reads "Reading…", and the live pill reads
 * "Reading the sign…". Nothing about the recognition pipeline is simulated.
 */
@RunWith(AndroidJUnit4::class)
class ProcessingStateRenderTest {

    private val instrumentation
        get() = InstrumentationRegistry.getInstrumentation()

    private fun declared(field: String): java.lang.reflect.Field =
        CameraScreen::class.java.getDeclaredField(field).apply { isAccessible = true }

    private fun settled(screen: CameraScreen): Boolean {
        if (!declared("modelReady").getBoolean(screen)) return false
        if (declared("camera").get(screen) == null) return false
        return declared("currentState").get(screen)?.toString() == "WAITING"
    }

    @Test
    fun processingStateRendersItsOwnAffordances() {
        instrumentation.uiAutomation.grantRuntimePermission(
            instrumentation.targetContext.packageName,
            android.Manifest.permission.CAMERA
        )

        val intent = Intent(Intent.ACTION_MAIN)
            .setClassName(instrumentation.targetContext, "org.echogesture.finalapp.MainActivity")
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        val activity = instrumentation.startActivitySync(intent) as MainActivity

        try {
            val screen = MainActivity::class.java.getDeclaredField("cameraScreen")
                .apply { isAccessible = true }.get(activity) as CameraScreen

            val deadline = System.currentTimeMillis() + 40_000
            var stable = 0
            while (stable < 6) {
                if (System.currentTimeMillis() > deadline) {
                    throw AssertionError("MainActivity never settled into a waiting state")
                }
                stable = if (settled(screen)) stable + 1 else 0
                Thread.sleep(250)
            }
            instrumentation.waitForIdleSync()

            val uiState = Class.forName("org.echogesture.finalapp.CameraScreen\$UiState")
            val processing = uiState.getDeclaredField("PROCESSING").get(null)
            val render = CameraScreen::class.java.getDeclaredMethod(
                "render", uiState, String::class.java, java.lang.Boolean.TYPE
            ).apply { isAccessible = true }

            instrumentation.runOnMainSync {
                render.invoke(screen, processing, null, false)
            }
            instrumentation.waitForIdleSync()
            Thread.sleep(300)

            val button = activity.findViewById<Button>(R.id.captureButton)
            assertEquals("button copies the processing action", "Reading…", button.text.toString())
            assertFalse("capture must be disabled while processing", button.isEnabled)

            val live = activity.findViewById<TextView>(R.id.liveStatus)
            assertEquals(
                "live pill names the processing state",
                activity.getString(R.string.camera_state_processing),
                live.text.toString()
            )

            val dir: File = instrumentation.targetContext.getExternalFilesDir(null)!!
            val ready = File(dir, "eg_processing_ready")
            val go = File(dir, "eg_processing_shot_done")
            ready.writeText("processing\n")
            val shotDeadline = System.currentTimeMillis() + 90_000
            while (!go.exists() && System.currentTimeMillis() < shotDeadline) Thread.sleep(250)
            ready.delete()
        } finally {
            instrumentation.runOnMainSync { activity.finish() }
        }
    }
}