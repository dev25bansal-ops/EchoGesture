package org.echogesture.finalapp

import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.android.material.bottomnavigation.BottomNavigationView
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The Camera tab must stop working when it is not on screen.
 *
 * Before this was fixed, `MainActivity.showTab` removed the Camera view but
 * never detached it: the CameraX analyser kept receiving frames and running
 * MediaPipe on them for Glossary, History and Settings. The user paid CPU and
 * battery for a screen they were not looking at.
 *
 * These tests pin the behaviour with a counter the analyser increments, rather
 * than trusting that an `unbindAll()` call is present.
 */
@RunWith(AndroidJUnit4::class)
class CameraLifecycleTest {

    private val instrumentation
        get() = InstrumentationRegistry.getInstrumentation()

    private fun cameraScreenOf(activity: MainActivity): CameraScreen {
        val f = MainActivity::class.java.getDeclaredField("cameraScreen")
            .apply { isAccessible = true }
        return f.get(activity) as CameraScreen
    }

    private fun navOf(activity: MainActivity): BottomNavigationView =
        activity.findViewById(R.id.bottomNav)

    /** Wait until the camera has bound and frames are actually arriving. */
    private fun awaitFrames(screen: CameraScreen, deadlineMs: Long = 20_000): Boolean {
        val deadline = SystemClock.elapsedRealtime() + deadlineMs
        while (SystemClock.elapsedRealtime() < deadline) {
            if (screen.analysedFrameCount > 0) return true
            Thread.sleep(100)
        }
        return false
    }

    /**
     * Leaving the Camera tab must stop the analyser. This is the actual
     * battery claim, measured rather than asserted.
     */
    @Test
    fun leavingTheCameraTabStopsTheAnalyser() {
        val activity = ActivityLaunch.fresh()
        try {
            val screen = cameraScreenOf(activity)
            assertTrue(
                "camera never produced a frame; cannot test teardown",
                awaitFrames(screen)
            )
            instrumentation.waitForIdleSync()

            // Let frames accumulate so the later comparison is meaningful.
            Thread.sleep(700)
            val before = screen.analysedFrameCount
            assertTrue("expected frames while on Camera, got $before", before > 0)

            instrumentation.runOnMainSync {
                navOf(activity).selectedItemId = R.id.nav_glossary
            }
            instrumentation.waitForIdleSync()

            assertTrue(
                "analyser should be detached off-tab",
                !screen.analyserAttached
            )

            // Give a still-running analyser ample time to be caught.
            Thread.sleep(1_500)
            val after = screen.analysedFrameCount
            assertEquals(
                "analyser kept consuming frames on the Glossary tab: " +
                    "$before -> $after",
                before,
                after
            )
        } finally {
            instrumentation.runOnMainSync { activity.finish() }
        }
    }

    /**
     * Coming back must restore a working camera and actually analyse again.
     * A teardown fix that leaves the tab dead is not a fix.
     */
    @Test
    fun returningToCameraRebindsAndAnalyses() {
        val activity = ActivityLaunch.fresh()
        try {
            val screen = cameraScreenOf(activity)
            assertTrue("camera never produced a frame", awaitFrames(screen))

            instrumentation.runOnMainSync {
                navOf(activity).selectedItemId = R.id.nav_glossary
            }
            instrumentation.waitForIdleSync()
            assertTrue("did not detach on leaving", !screen.analyserAttached)

            val atUnbind = screen.analysedFrameCount

            instrumentation.runOnMainSync {
                navOf(activity).selectedItemId = R.id.nav_camera
            }
            instrumentation.waitForIdleSync()

            val deadline = SystemClock.elapsedRealtime() + 20_000
            var resumed = false
            while (SystemClock.elapsedRealtime() < deadline) {
                if (screen.analysedFrameCount > atUnbind) { resumed = true; break }
                Thread.sleep(100)
            }
            assertTrue(
                "camera did not resume analysing after returning " +
                    "(count stuck at $atUnbind)",
                resumed
            )
            assertTrue("analyser should be attached again", screen.analyserAttached)
        } finally {
            instrumentation.runOnMainSync { activity.finish() }
        }
    }

    /**
     * Re-binding the tab must not make the app look like it reloaded the
     * models. The footer carries the ensemble identity and the model hash; if
     * a tab round trip reset it to "Preparing camera", the user would
     * reasonably think the nine models were re-initialised every time.
     */
    @Test
    fun returningToCameraKeepsTheNineModelFooter() {
        val activity = ActivityLaunch.fresh()
        try {
            val screen = cameraScreenOf(activity)
            assertTrue("camera never produced a frame", awaitFrames(screen))

            val footer = screen.root.findViewById<android.widget.TextView>(R.id.modelIdentity)
            val deadline = SystemClock.elapsedRealtime() + 20_000
            while (SystemClock.elapsedRealtime() < deadline &&
                !footer.text.contains("9-model")
            ) {
                Thread.sleep(100)
            }
            val before = footer.text.toString()
            assertTrue("footer should name the ensemble, was '$before'", before.contains("9-model"))

            instrumentation.runOnMainSync {
                navOf(activity).selectedItemId = R.id.nav_glossary
            }
            instrumentation.waitForIdleSync()
            Thread.sleep(300)

            instrumentation.runOnMainSync {
                navOf(activity).selectedItemId = R.id.nav_camera
            }
            instrumentation.waitForIdleSync()

            val after = footer.text.toString()
            assertEquals(
                "footer changed across a tab round trip; models were not reloaded",
                before,
                after
            )
        } finally {
            instrumentation.runOnMainSync { activity.finish() }
        }
    }

    /**
     * A recording abandoned by navigating away must not survive to be
     * classified on some later visit. This is the "stale recording" case: the
     * user taps Start, leaves, comes back, taps Finish, and the app would
     * otherwise classify an interval containing a tab change they never
     * signed through.
     */
    @Test
    fun navigatingAwayCancelsAnInFlightRecording() {
        val activity = ActivityLaunch.fresh()
        try {
            val screen = cameraScreenOf(activity)
            assertTrue("camera never produced a frame", awaitFrames(screen))
            instrumentation.waitForIdleSync()

            val trackerField = CameraScreen::class.java.getDeclaredField("tracker")
                .apply { isAccessible = true }
            val tracker = trackerField.get(screen) as HandTracker

            instrumentation.runOnMainSync {
                screen.root.findViewById<android.view.View>(R.id.captureButton).performClick()
            }
            instrumentation.waitForIdleSync()
            Thread.sleep(400)
            assertTrue("recording did not start", tracker.isRecording())

            instrumentation.runOnMainSync {
                navOf(activity).selectedItemId = R.id.nav_history
            }
            instrumentation.waitForIdleSync()

            assertTrue(
                "an abandoned recording must not stay armed after leaving the tab",
                !tracker.isRecording()
            )
        } finally {
            instrumentation.runOnMainSync { activity.finish() }
        }
    }

    /**
     * Repeating the round trip must not leak: the analyser has to keep
     * starting and stopping cleanly, and frames must keep advancing on each
     * return rather than the tab degrading into a dead preview.
     */
    @Test
    fun repeatedTabRoundTripsKeepTheCameraUsable() {
        val activity = ActivityLaunch.fresh()
        try {
            val screen = cameraScreenOf(activity)
            assertTrue("camera never produced a frame", awaitFrames(screen))

            repeat(3) { i ->
                val beforeLeave = screen.analysedFrameCount
                instrumentation.runOnMainSync {
                    navOf(activity).selectedItemId = R.id.nav_settings
                }
                instrumentation.waitForIdleSync()
                Thread.sleep(400)
                assertEquals(
                    "round $i: analyser ran while off-tab",
                    beforeLeave,
                    screen.analysedFrameCount
                )

                instrumentation.runOnMainSync {
                    navOf(activity).selectedItemId = R.id.nav_camera
                }
                instrumentation.waitForIdleSync()

                val deadline = SystemClock.elapsedRealtime() + 20_000
                var resumed = false
                while (SystemClock.elapsedRealtime() < deadline) {
                    if (screen.analysedFrameCount > beforeLeave) { resumed = true; break }
                    Thread.sleep(100)
                }
                assertTrue("round $i: camera did not resume", resumed)
            }
        } finally {
            instrumentation.runOnMainSync { activity.finish() }
        }
    }

    /**
     * Quantifies the saving rather than just asserting it. Reports how many
     * analyser callbacks would have run during a five-second stay on another
     * tab, which is the work the fix removes per tab visit.
     */
    @Test
    fun measuresAnalyserWorkSavedWhileOffTab() {
        val activity = ActivityLaunch.fresh()
        try {
            val screen = cameraScreenOf(activity)
            assertTrue("camera never produced a frame", awaitFrames(screen))

            // Frame rate while the Camera tab is actually visible.
            instrumentation.waitForIdleSync()
            val t0 = SystemClock.elapsedRealtime()
            val f0 = screen.analysedFrameCount
            Thread.sleep(2_000)
            val f1 = screen.analysedFrameCount
            val t1 = SystemClock.elapsedRealtime()
            val fpsVisible = (f1 - f0).toDouble() / ((t1 - t0) / 1000.0)

            instrumentation.runOnMainSync {
                navOf(activity).selectedItemId = R.id.nav_glossary
            }
            instrumentation.waitForIdleSync()

            val g0 = screen.analysedFrameCount
            Thread.sleep(5_000)
            val g1 = screen.analysedFrameCount
            val offTabFrames = g1 - g0

            android.util.Log.i(
                "CameraLifecycleTest",
                "fps_visible=%.1f off_tab_frames_in_5s=%d".format(fpsVisible, offTabFrames)
            )
            assertTrue(
                "camera should be analysing while visible, saw %.1f fps".format(fpsVisible),
                fpsVisible > 1.0
            )
            assertEquals(
                "analyser must do no work for a tab the user is not looking at",
                0L,
                offTabFrames
            )
        } finally {
            instrumentation.runOnMainSync { activity.finish() }
        }
    }
}
