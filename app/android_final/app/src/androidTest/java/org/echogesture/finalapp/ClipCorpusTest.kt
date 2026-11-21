package org.echogesture.finalapp

import android.net.Uri
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Locale

/** Measures the file-picker's real decoder, detector, feature builder and classifier together. */
@RunWith(AndroidJUnit4::class)
class ClipCorpusTest {
    @Test
    fun benchmarkProductionFileRecognition() {
        val runner = InstrumentationRegistry.getInstrumentation()
        val context = runner.targetContext
        val directory = File(context.filesDir, "file-corpus")
        val manifest = File(directory, "manifest.tsv")
        val required = InstrumentationRegistry.getArguments().getString("requireCorpus") == "true"
        if (required) check(manifest.isFile) { "Push the labelled corpus to file-corpus before running" }
        assumeTrue(manifest.isFile)
        val clipLabels = manifest.readLines().filter { it.isNotBlank() }.map { line ->
            val (clipName, gloss) = line.split('\t', limit = 2)
            require(File(clipName).name == clipName)
            File(directory, clipName) to gloss
        }
        if (required) check(clipLabels.size >= 30) { "Run the complete labelled sample" }
        val loaded = ModelContract.load(context.assets)
        val outcomes = mutableListOf<String>()
        ClipAnalyzer(context).use { analyzer ->
            LandmarkClassifier.create(loaded).use { classifier ->
                for ((clip, expected) in clipLabels) {
                    val started = SystemClock.elapsedRealtime()
                    val recording = analyzer.analyze(Uri.fromFile(clip))
                    val frames = recording.frames
                    val coverage = frames.count { it.hands.isNotEmpty() }.toFloat() / frames.size
                    if (coverage < 0.15f) {
                        outcomes += "$expected,,0,0,false,${SystemClock.elapsedRealtime() - started}"
                        continue
                    }
                    val features = RecordingFeatures.buildBoth(frames)
                    val result = classifier.predict(features.featuresXY, features.mask, features.featuresXYZ)
                    val duration = SystemClock.elapsedRealtime() - started
                    val score = result.confidence * 100
                    val accepted = ResultPolicy.isAccepted(result.margin, score, fromClip = true)
                    outcomes += String.format(
                        Locale.US, "%s,%s,%.6f,%.6f,%s,%d",
                        expected, result.prediction, score, result.margin, accepted, duration
                    )
                }
            }
        }
        assertEquals(clipLabels.size, outcomes.size)
        File(directory, "results.csv").writeText(
            "expected,prediction,score_percent,margin,accepted,wall_ms\n" + outcomes.joinToString("\n") + "\n"
        )
    }
}
