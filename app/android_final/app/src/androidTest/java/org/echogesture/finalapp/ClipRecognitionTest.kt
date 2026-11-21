package org.echogesture.finalapp

import android.net.Uri
import android.os.SystemClock
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class ClipRecognitionTest {
    @Test
    fun selectedClipRecoversAfterBadInputAndCanBeReplayed() {
        val runner = InstrumentationRegistry.getInstrumentation()
        val dir = runner.targetContext.getExternalFilesDir(null)!!
        val clip = File(dir, "review-what-for.mp4")
        if (InstrumentationRegistry.getArguments().getString("requireClip") == "true") {
            check(clip.isFile) { "Push review-what-for.mp4 to the app's external files directory" }
        }
        assumeTrue(clip.isFile)

        val activity = ActivityLaunch.fresh()
        val broken = File(activity.cacheDir, "invalid.mp4").apply { writeText("not a video") }
        try {
            val screen = MainActivity::class.java.getDeclaredField("cameraScreen").apply {
                isAccessible = true
            }.get(activity)
            val analyze = CameraScreen::class.java.getDeclaredMethod("analyzeClip", Uri::class.java).apply {
                isAccessible = true
            }
            waitFor {
                activity.findViewById<Button>(R.id.clipButton).isEnabled
            }
            runner.runOnMainSync { analyze.invoke(screen, Uri.fromFile(broken)) }
            waitFor {
                activity.findViewById<TextView>(R.id.statusDetail).text.toString() ==
                    activity.getString(R.string.clip_unreadable)
            }

            repeat(2) {
                runner.runOnMainSync { analyze.invoke(screen, Uri.fromFile(clip)) }
                waitFor { activity.findViewById<View>(R.id.resultCard).visibility == View.VISIBLE }
                assertEquals("WHATFOR1", activity.findViewById<TextView>(R.id.captionText).text.toString())
            }
        } finally {
            broken.delete()
            runner.runOnMainSync { activity.finish() }
        }
    }

    private fun waitFor(predicate: () -> Boolean) {
        val runner = InstrumentationRegistry.getInstrumentation()
        val deadline = SystemClock.elapsedRealtime() + 45_000
        while (SystemClock.elapsedRealtime() < deadline) {
            var matched = false
            runner.runOnMainSync { matched = predicate() }
            if (matched) return
            SystemClock.sleep(100)
        }
        fail("Timed out waiting for the clip recognition screen")
    }
}
