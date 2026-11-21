package org.echogesture.finalapp

import android.os.SystemClock
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.android.material.bottomnavigation.BottomNavigationView
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CameraLifecycleTest {
    @Test
    fun bothLensesAnalyseLessWhileIdleAndKeepCapturingAfterSwitch() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("requirePacing") == "true")
        val runner = InstrumentationRegistry.getInstrumentation()
        val activity = ActivityLaunch.fresh()
        try {
            val capture = activity.findViewById<Button>(R.id.captureButton)
            val flip = activity.findViewById<View>(R.id.flipButton)
            val screen = MainActivity::class.java.getDeclaredField("cameraScreen")
                .apply { isAccessible = true }.get(activity)
            val camera = CameraScreen::class.java.getDeclaredField("camera").apply { isAccessible = true }
            val trackerField = CameraScreen::class.java.getDeclaredField("tracker").apply { isAccessible = true }
            val tracking = CameraScreen::class.java.getDeclaredField("currentState").apply { isAccessible = true }
            waitFor { capture.isEnabled }
            val tracker = trackerField.get(screen) as HandTracker
            val lensCounts = mutableMapOf<Int, Pair<Int, Int>>()
            repeat(2) {
                waitFor { capture.isEnabled && tracker.snapshot().size >= 4 }
                var lens = -1
                runner.runOnMainSync { lens = (camera.get(screen) as Camera).cameraInfo.lensFacing }
                val idle = countFrames(tracker)
                runner.runOnMainSync { capture.performClick() }
                waitFor { tracking.get(screen)?.toString() == "AUTO_ARMED" }
                val armed = countFrames(tracker)
                lensCounts[lens] = idle to armed
                println("CAMERA_PACING lens=$lens idle=$idle armed=$armed window_ms=2000")
                runner.runOnMainSync { flip.performClick() }
                waitFor {
                    capture.isEnabled && capture.text.toString() == activity.getString(R.string.start_recognition)
                }
            }
            assertTrue(CameraSelector.LENS_FACING_FRONT in lensCounts)
            assertTrue(CameraSelector.LENS_FACING_BACK in lensCounts)
            for ((lens, counts) in lensCounts) {
                val (idle, armed) = counts
                assertTrue("lens=$lens idle=$idle must stay at about 8 fps", idle in 1..17)
                assertTrue("lens=$lens armed=$armed should run at full rate", armed > idle)
            }
        } finally {
            runner.runOnMainSync { activity.finish() }
        }
    }

    @Test
    fun oneStartTapShowsTheResultAfterHandsLeave() {
        val runner = InstrumentationRegistry.getInstrumentation()
        val activity = ActivityLaunch.fresh()
        try {
            val capture = activity.findViewById<Button>(R.id.captureButton)
            val card = activity.findViewById<View>(R.id.resultCard)
            val screen = MainActivity::class.java.getDeclaredField("cameraScreen")
                .apply { isAccessible = true }.get(activity)
            val trackerField = CameraScreen::class.java.getDeclaredField("tracker")
                .apply { isAccessible = true }
            waitFor { capture.isEnabled }
            val tracker = trackerField.get(screen) as HandTracker
            runner.runOnMainSync { capture.performClick() }
            val start = SystemClock.elapsedRealtime()
            // Supply landmarks to the live tracker. The camera analyzer still
            // drives state changes, inference and delivery of the result card.
            for (i in 1..36) {
                val hand = HandTracker.Hand(1, FloatArray(63) { j ->
                    if (j < 3) 0f else (j % 7 + i % 2) * 0.02f
                })
                tracker.offerForTest(HandTracker.Frame(listOf(hand), start + i * 40L))
            }
            waitFor { !capture.isEnabled && capture.text == activity.getString(R.string.auto_signing_status) }
            for (i in 37..62) {
                tracker.offerForTest(HandTracker.Frame(emptyList(), start + i * 40L))
            }
            waitFor {
                card.visibility == View.VISIBLE && capture.isEnabled &&
                    capture.text == activity.getString(R.string.action_recognise_again)
            }
        } finally {
            runner.runOnMainSync { activity.finish() }
        }
    }

    private fun countFrames(tracker: HandTracker): Int {
        val start = SystemClock.elapsedRealtime()
        SystemClock.sleep(2000)
        return tracker.snapshot().count { it.capturedAtMs in start until (start + 2000) }
    }

    @Test
    fun emptyPreviewTimesOutAndTabSwitchDiscardsAnotherAttempt() {
        val runner = InstrumentationRegistry.getInstrumentation()
        val activity = ActivityLaunch.fresh()
        try {
            val button = activity.findViewById<Button>(R.id.captureButton)
            val status = activity.findViewById<TextView>(R.id.statusDetail)
            val card = activity.findViewById<View>(R.id.resultCard)
            waitFor { button.isEnabled }
            runner.runOnMainSync { button.performClick() }
            waitFor { button.text.toString() == activity.getString(R.string.auto_cancel) }
            waitFor {
                status.text.toString() == activity.getString(R.string.auto_no_sign) &&
                    card.visibility != View.VISIBLE && button.isEnabled
            }

            runner.runOnMainSync {
                button.performClick()
                val tabs = activity.findViewById<BottomNavigationView>(R.id.bottomNav)
                tabs.selectedItemId = R.id.nav_settings
                tabs.selectedItemId = R.id.nav_camera
            }
            waitFor {
                button.isEnabled && button.text.toString() == activity.getString(R.string.start_recognition)
            }
        } finally {
            runner.runOnMainSync { activity.finish() }
        }
    }

    private fun waitFor(condition: () -> Boolean) {
        val runner = InstrumentationRegistry.getInstrumentation()
        val deadline = SystemClock.elapsedRealtime() + 20_000L
        while (SystemClock.elapsedRealtime() < deadline) {
            var matched = false
            runner.runOnMainSync { matched = condition() }
            if (matched) return
            SystemClock.sleep(100)
        }
        fail("Camera screen did not reach the expected state")
    }
}
