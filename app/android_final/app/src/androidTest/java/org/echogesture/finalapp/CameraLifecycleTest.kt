package org.echogesture.finalapp

import android.os.SystemClock
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.android.material.bottomnavigation.BottomNavigationView
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CameraLifecycleTest {
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
