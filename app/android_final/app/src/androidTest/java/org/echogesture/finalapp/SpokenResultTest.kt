package org.echogesture.finalapp

import android.media.MediaPlayer
import android.os.SystemClock
import android.widget.Button
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SpokenResultTest {
    @Test
    fun confirmedResultPlaysOfflineAndSpeechSettingCanMuteIt() {
        val runner = InstrumentationRegistry.getInstrumentation()
        val activity = ActivityLaunch.fresh()
        val prefs = AppPreferences(activity)
        val oldSpeech = prefs.speech
        val cameraScreen = MainActivity::class.java.getDeclaredField("cameraScreen")
            .apply { isAccessible = true }.get(activity) as CameraScreen
        val speaker = MainActivity::class.java.getDeclaredField("speaker")
            .apply { isAccessible = true }.get(activity) as Speaker
        val player = Speaker::class.java.getDeclaredField("player").apply { isAccessible = true }
        val tracker = CameraScreen::class.java.getDeclaredField("tracker").apply { isAccessible = true }
        val render = CameraScreen::class.java.getDeclaredMethod(
            "renderResult",
            LandmarkClassifier.Result::class.java,
            java.lang.Long.TYPE,
            HandTracker.Capture::class.java
        ).apply { isAccessible = true }
        val result = LandmarkClassifier.Result(
            "WHATFOR1", listOf("WHATFOR1", "DEAF1"), .96f, .92f, 4L
        )
        val capture = HandTracker.Capture(emptyList(), 0L, 0f, 0)

        fun await(condition: () -> Boolean): Boolean {
            val deadline = SystemClock.elapsedRealtime() + 15_000L
            while (SystemClock.elapsedRealtime() < deadline) {
                var ready = false
                runner.runOnMainSync { ready = condition() }
                if (ready) return true
                SystemClock.sleep(25)
            }
            return false
        }

        fun show(reading: LandmarkClassifier.Result) {
            runner.runOnMainSync { render.invoke(cameraScreen, reading, 5L, capture) }
        }

        try {
            assertTrue(await { activity.findViewById<Button>(R.id.captureButton).isEnabled })
            val handTracker = tracker.get(cameraScreen) as HandTracker
            prefs.speech = true
            show(result)
            assertTrue("The confirmed caption should play its bundled audio", await {
                runCatching { (player.get(speaker) as? MediaPlayer)?.isPlaying == true }.getOrDefault(false)
            })
            assertTrue("Playback releases its player when finished", await { player.get(speaker) == null })

            val frames = handTracker.snapshot()
            SystemClock.sleep(450)
            assertEquals("The detector must stay idle while displaying a result", frames, handTracker.snapshot())

            show(result.copy(confidence = .20f, margin = .02f))
            runner.runOnMainSync { assertFalse(player.get(speaker) is MediaPlayer) }
            val fullRender = CameraScreen::class.java.getDeclaredMethod(
                "renderResult", LandmarkClassifier.Result::class.java, java.lang.Long.TYPE,
                HandTracker.Capture::class.java, java.lang.Boolean.TYPE, java.lang.Boolean.TYPE,
                java.lang.Boolean.TYPE
            ).apply { isAccessible = true }
            val moderate = result.copy(confidence = .58f, margin = .42f, agreeingMembers = 9)
            runner.runOnMainSync {
                fullRender.invoke(cameraScreen, moderate, 15L, capture, false, true, true)
                assertTrue(activity.findViewById<Button>(R.id.speakButton).isEnabled)
                val badge = activity.findViewById<android.widget.TextView>(R.id.confidenceBadge)
                assertEquals(activity.getString(R.string.result_score, "%.1f".format(58f)), badge.text.toString())
            }
            assertTrue(await {
                runCatching { (player.get(speaker) as? MediaPlayer)?.isPlaying == true }.getOrDefault(false)
            })
            assertTrue(await { player.get(speaker) == null })
            show(result.copy(confidence = .48f, margin = .02f))
            runner.runOnMainSync {
                assertFalse(activity.findViewById<Button>(R.id.speakButton).isEnabled)
                val options = activity.findViewById<android.widget.TextView>(R.id.alternativesText)
                assertTrue(options.text.contains(activity.getString(R.string.result_runner_up, "DEAF1", 46f)))
            }
            prefs.speech = false
            show(result)
            runner.runOnMainSync { assertFalse(player.get(speaker) is MediaPlayer) }
        } finally {
            runner.runOnMainSync {
                prefs.speech = oldSpeech
                activity.finish()
            }
        }
    }
}
