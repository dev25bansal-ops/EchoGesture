package org.echogesture.finalapp

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Rect
import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.camera.core.ImageProxy
import androidx.camera.core.ImmutableImageInfo
import androidx.camera.core.impl.TagBundle
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.lang.reflect.Proxy

@RunWith(AndroidJUnit4::class)
class PortraitRecognitionTest {
    @Test
    fun portraitCaptureImprovesRecognitionWithCameraRotation() {
        val runner = InstrumentationRegistry.getInstrumentation()
        val context = runner.targetContext
        val directory = File(context.filesDir, "file-corpus")
        val manifest = File(directory, "portrait.tsv")
        val required = InstrumentationRegistry.getArguments().getString("requirePortrait") == "true"
        if (required) check(manifest.isFile) { "Stage the portrait replay clips first" }
        assumeTrue(manifest.isFile)
        val clips = manifest.readLines().filter { it.isNotBlank() }
        if (required) check(clips.size == 60) { "Stage all 60 portrait replay clips" }
        val rows = mutableListOf("expected,legacy,legacy_score,legacy_accepted,current,current_score,current_accepted")
        LandmarkClassifier.create(ModelContract.load(context.assets)).use { classifier ->
            for ((index, line) in clips.withIndex()) {
                val (clip, expected) = line.split('\t')
                require(File(clip).name == clip)
                val file = File(directory, clip)
                val uri = Uri.fromFile(file)
                val retriever = MediaMetadataRetriever()
                val durationMs = try {
                    retriever.setDataSource(file.path)
                    checkNotNull(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)).toLong()
                } finally {
                    retriever.release()
                }
                val rotation = if (index % 2 == 0) 90 else 270
                val portrait = mutableListOf<HandTracker.Frame>()
                HandTracker.create(context, ModelContract.DETECTOR_ASSET).use { tracker ->
                    SequentialClipDecoder.sample(context, uri, durationMs, 32, 640, { true }) { bitmap, ms ->
                        val tall = Bitmap.createBitmap(480, 640, Bitmap.Config.ARGB_8888)
                        val h = 480 * bitmap.height / bitmap.width
                        val top = (640 - h) / 2
                        try {
                            Canvas(tall).apply {
                                drawColor(Color.DKGRAY)
                                drawBitmap(bitmap, null, Rect(0, top, 480, top + h), null)
                            }
                            val sensor = Bitmap.createBitmap(
                                tall, 0, 0, tall.width, tall.height,
                                Matrix().apply { postRotate(-rotation.toFloat()) }, false
                            )
                            portrait += replay(tracker, sensor, ms, rotation)
                        } finally {
                            tall.recycle()
                            bitmap.recycle()
                        }
                    }
                }
                assertTrue("Expected enough frames for $clip", portrait.size >= 28)
                // Undo 480x640 aspect correction to replay the same observed
                // landmarks with V5.7 coordinates and identical timestamps.
                val legacy = portrait.map { frame ->
                    frame.copy(hands = frame.hands.map { hand ->
                        hand.copy(xyz = hand.xyz.copyOf().also { xyz ->
                            for (i in 1 until xyz.size step 3) xyz[i] *= 9f / 16f
                        })
                    })
                }
                rows += "$expected,${auto(classifier, legacy)},${auto(classifier, portrait)}"
            }
        }
        File(directory, "portrait-results.csv").writeText(rows.joinToString("\n") + "\n")
        if (required) {
            val values = rows.drop(1).map { it.split(',') }
            val legacyCorrect = values.count { it[0] == it[1] }
            val currentCorrect = values.count { it[0] == it[4] }
            val legacyAccepted = values.count { it[0] == it[1] && it[3] == "true" }
            val currentAccepted = values.count { it[0] == it[4] && it[6] == "true" }
            assertTrue("Portrait top match: $legacyCorrect to $currentCorrect", currentCorrect > legacyCorrect)
            assertTrue("Correct accepted captions: $legacyAccepted to $currentAccepted",
                currentAccepted > legacyAccepted)
            assertTrue("Unexpected accepted caption on the pinned replay set",
                values.none { it[6] == "true" && it[4] != it[0] })
        }
    }

    private fun replay(
        tracker: HandTracker, bitmap: Bitmap, timeMs: Long, rotation: Int
    ): HandTracker.Frame {
        val info = ImmutableImageInfo.create(TagBundle.emptyBundle(), timeMs * 1_000_000, rotation, Matrix())
        val proxy = Proxy.newProxyInstance(
            ImageProxy::class.java.classLoader, arrayOf(ImageProxy::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "toBitmap" -> bitmap
                "getImageInfo" -> info
                "close" -> null
                else -> error("Unexpected ImageProxy access: ${method.name}")
            }
        } as ImageProxy
        try {
            check(tracker.analyse(proxy)) { "Camera frame dropped" }
            return tracker.snapshot().last().copy(capturedAtMs = timeMs)
        } finally {
            if (!bitmap.isRecycled) bitmap.recycle()
        }
    }

    private fun auto(classifier: LandmarkClassifier, frames: List<HandTracker.Frame>): String {
        val s = AutoSegmenter()
        s.arm(0)
        val lead = 600L
        val inputs = frames.map { it.copy(capturedAtMs = it.capturedAtMs + lead) }
        val tail = inputs.last().capturedAtMs
        val result = (listOf(150L, 300L, 450L).map { HandTracker.Frame(emptyList(), it) } +
            inputs + (100L..1200L step 100L).map { HandTracker.Frame(emptyList(), tail + it) })
            .firstNotNullOfOrNull(s::offer)
        return if (result?.isUsable == true) classify(classifier, result.frames) else ",0,false"
    }

    private fun classify(classifier: LandmarkClassifier, frames: List<HandTracker.Frame>): String {
        if (frames.none { it.hands.isNotEmpty() }) return ",0,false"
        val recording = SignRecorder.Recording(frames, frames.first().capturedAtMs, frames.last().capturedAtMs)
        val features = RecordingFeatures.buildBoth(recording)
        val p = classifier.predict(features.featuresXY, features.mask, features.featuresXYZ)
        val accepted = ResultPolicy.isAccepted(p.margin, p.confidence * 100f)
        return "${p.prediction},${p.confidence},$accepted"
    }
}
