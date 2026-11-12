package org.echogesture.finalapp

import android.Manifest
import android.content.Intent
import androidx.test.platform.app.InstrumentationRegistry

/**
 * Launches MainActivity in a state a test can rely on.
 *
 * A previous test's activity, or an app a developer left open, keeps the camera
 * open for the life of its process. The new activity's CameraX bind then fails
 * and it never reaches WAITING, so a test that waits for a settled camera times
 * out and reports a failure that has nothing to do with the code under test.
 *
 * The dead activity is reaped with `am kill`, NOT `am force-stop`. The
 * instrumentation is injected into this same package, so force-stop would kill
 * the test runner along with it; `am kill` reaps the background process and
 * leaves the foreground instrumentation alone.
 */
object ActivityLaunch {

    private fun shell(cmd: String) {
        val pfd = InstrumentationRegistry.getInstrumentation()
            .uiAutomation.executeShellCommand(cmd)
        java.io.FileInputStream(pfd.fileDescriptor).use { it.readBytes() }
    }

    fun fresh(): MainActivity {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        // Best effort: release any camera held by a previous activity in this
        // package. Harmless when nothing is running.
        shell("am kill ${instrumentation.targetContext.packageName}")
        Thread.sleep(500)

        instrumentation.uiAutomation.grantRuntimePermission(
            instrumentation.targetContext.packageName,
            Manifest.permission.CAMERA
        )

        val intent = Intent(Intent.ACTION_MAIN)
            .setClassName(instrumentation.targetContext, "org.echogesture.finalapp.MainActivity")
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        return instrumentation.startActivitySync(intent) as MainActivity
    }
}