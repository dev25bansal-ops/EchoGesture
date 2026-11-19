package org.echogesture.finalapp

import android.net.Uri
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Does the clip-upload path classify a known clip correctly?
 *
 * This is the deterministic self-test the camera path could not provide. On
 * 2026-11-19 a live capture of the NOON sign returned FOREIGNER1 at 0.51-0.60
 * while the same clip scored NOON1 at 0.9731 through Python and through this
 * app's own classifier. Frame rate was ruled out as the cause. A file has no
 * camera scene, so if the clip path is wired correctly it MUST reproduce the
 * Python answer.
 *
 * Asserting the gloss rather than printing it is deliberate: a drift here means
 * [ClipAnalyzer] and the camera path have diverged, which would make the
 * feature lie to the user.
 */
class ClipUploadTest {

    private fun report(line: String) {
        println(line)
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        runCatching {
            File(ctx.filesDir, "clip_upload.txt").appendText(line + "\n")
        }
    }

    @Test
    fun knownClipClassifiesAsItsOwnGloss() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        runCatching { File(ctx.filesDir, "clip_upload.txt").writeText("") }

        // Prefer the pushed corpus; fall back to app-private external files.
        val candidates = listOfNotNull(
            File("/sdcard/Movies/EchoGesture"),
            ctx.getExternalFilesDir(null),
            ctx.filesDir
        )
        val dir = candidates.firstOrNull { d ->
            d.listFiles { f -> f.isFile && f.name.contains("NOON", true) }?.isNotEmpty() == true
        }
        assumeTrue(
            "NOON clip not on device; copy it to /sdcard/Movies/EchoGesture. " +
                "This run is INCOMPLETE, not passing.",
            dir != null
        )
        val clip = dir!!.listFiles { f -> f.isFile && f.name.contains("NOON", true) }!!
            .sortedBy { it.name }.first()
        report("CLIP file=${clip.name} size=${clip.length()}")

        val outcome = ClipAnalyzer.analyse(ctx, Uri.fromFile(clip))
        report("CLIP outcome=${outcome.javaClass.simpleName}")
        when (outcome) {
            is ClipAnalyzer.Outcome.Failed -> {
                report("CLIP FAILED reason=${outcome.reason} detail=${outcome.detail}")
            }
            is ClipAnalyzer.Outcome.Analysed -> {
                report(
                    "CLIP analysed frames=${outcome.frameCount} " +
                        "hands=${outcome.handFrames} truncated=${outcome.truncated}"
                )
                val loaded = ModelContract.load(ctx.assets)
                report("CLIP model loaded members=${loaded.members.size}")
                val classifier = LandmarkClassifier.create(loaded)
                try {
                    val t = RecordingFeatures.buildBoth(outcome.recording)
                    val r = classifier.predict(t.featuresXY, t.mask, t.featuresXYZ)
                    report(
                        "CLIP pred=${r.prediction} conf=${"%.4f".format(r.confidence)} " +
                            "margin=${"%.4f".format(r.margin)} top5=${r.top5.joinToString("|")}"
                    )
                    assertTrue(
                        "expected NOON1 from the NOON clip, got ${r.prediction} " +
                            "at ${"%.4f".format(r.confidence)}; " +
                            "top5=${r.top5.joinToString("|")}",
                        r.prediction == "NOON1"
                    )
                } finally {
                    classifier.close()
                }
            }
        }
    }
}
