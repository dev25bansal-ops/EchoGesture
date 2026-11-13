package org.echogesture.finalapp

import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The pre-tap contract on a real device/emulator, against the shipped tracker.
 *
 * The pure decisions are covered by PreTapCaptureContractTest on the JVM; this
 * checks the parts that only exist on-device — constructing the tracker, the
 * real wall clock, and the ring under the monitor.
 */
class PreTapOnDeviceTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun aTapImmediatelyAfterStartupIsRefusedRatherThanAnsweredLate() {
        HandTracker.create(context, "hand_landmarker.task").use { t ->
            val started = SystemClock.elapsedRealtime()
            val r = t.beginCapture()
            val elapsed = SystemClock.elapsedRealtime() - started

            assertTrue(
                "the decision must be synchronous, took ${elapsed}ms",
                elapsed < 100
            )
            assertTrue(
                "an empty ring must be refused, got $r",
                r is HandTracker.CaptureRequest.Insufficient
            )
            assertEquals(
                HandTracker.CaptureRequest.Insufficient.Reason.NO_FRAMES,
                (r as HandTracker.CaptureRequest.Insufficient).reason
            )
        }
    }

    @Test
    fun clearingTheTrackerForgetsBothFramesAndTheRepeatGuard() {
        HandTracker.create(context, "hand_landmarker.task").use { t ->
            assertEquals(0, t.status().first)
            t.beginCapture()
            t.clear()
            assertEquals(0, t.status().first)
            val again = t.beginCapture()
            // Still NO_FRAMES, not REPEATED: the guard must not survive a clear.
            assertEquals(
                HandTracker.CaptureRequest.Insufficient.Reason.NO_FRAMES,
                (again as HandTracker.CaptureRequest.Insufficient).reason
            )
        }
    }
}
