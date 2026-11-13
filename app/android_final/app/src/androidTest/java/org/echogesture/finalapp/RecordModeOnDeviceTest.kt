package org.echogesture.finalapp

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Record mode against the shipped tracker, on a real device/emulator.
 *
 * The pure contract is covered by [SignRecorderTest] on the JVM. This checks
 * the parts that only exist on-device: that the tracker owns a recorder, that
 * starting one does not disturb the rolling ring quick scan depends on, and
 * that a camera switch ends a recording rather than splicing two sensors.
 */
@RunWith(AndroidJUnit4::class)
class RecordModeOnDeviceTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun startingARecordingLeavesQuickScanAloneAndProducesNoFramesYet() {
        HandTracker.create(context, ModelContract.DETECTOR_ASSET).use { t ->
            assertEquals(0, t.recordingFrameCount())
            assertFalse(t.isRecording())

            val start = android.os.SystemClock.elapsedRealtime()
            t.startRecording(start)

            assertTrue(t.isRecording())
            assertEquals(start, t.recordingStartedAt())
            assertEquals("a new recording starts empty", 0, t.recordingFrameCount())
            // Quick scan's ring is untouched: it is a separate buffer.
            assertEquals(0, t.status().first)
            assertFalse("a fresh recording must not auto-stop immediately",
                t.shouldAutoStopRecording(start))
        }
    }

    @Test
    fun stoppingWithoutAnyFramesIsReportedRatherThanClassified() {
        HandTracker.create(context, ModelContract.DETECTOR_ASSET).use { t ->
            val start = android.os.SystemClock.elapsedRealtime()
            t.startRecording(start)
            val outcome = t.stopRecording(start + 2_000)
            assertTrue("must be Unusable", outcome is SignRecorder.Result.Unusable)
            assertEquals(
                SignRecorder.Rejection.NO_FRAMES,
                (outcome as SignRecorder.Result.Unusable).reason
            )
            assertFalse(t.isRecording())
        }
    }

    @Test
    fun aCameraSwitchEndsTheRecordingInsteadOfMixingSensors() {
        HandTracker.create(context, ModelContract.DETECTOR_ASSET).use { t ->
            t.startRecording(android.os.SystemClock.elapsedRealtime())
            assertTrue(t.isRecording())
            // switchCamera() calls clear().
            t.clear()
            assertFalse("a camera switch must end the recording", t.isRecording())
            val outcome = t.stopRecording(android.os.SystemClock.elapsedRealtime())
            assertEquals(
                SignRecorder.Rejection.NOT_STARTED,
                (outcome as SignRecorder.Result.Unusable).reason
            )
        }
    }

    @Test
    fun cancellingLeavesNoRecordingBehind() {
        HandTracker.create(context, ModelContract.DETECTOR_ASSET).use { t ->
            t.startRecording(android.os.SystemClock.elapsedRealtime())
            t.cancelRecording()
            assertFalse(t.isRecording())
            assertEquals(0, t.recordingFrameCount())
        }
    }
}