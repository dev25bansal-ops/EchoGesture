package org.echogesture.finalapp

import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.camera.core.ImageProxy
import androidx.camera.core.ImmutableImageInfo
import androidx.camera.core.impl.TagBundle
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.lang.reflect.Proxy

/** Replays known signs through the real camera analyser with CameraX rotation metadata. */
@RunWith(AndroidJUnit4::class)
class CameraRotationReplayTest {
    @Test
    fun cameraBufferRotationPreservesSignPredictions() {
        val runner = InstrumentationRegistry.getInstrumentation()
        val context = runner.targetContext
        val directory = File(context.filesDir, "file-corpus")
        val manifest = File(directory, "camera.tsv")
        if (InstrumentationRegistry.getArguments().getString("requireCameraCorpus") == "true") {
            check(manifest.isFile) { "Stage the camera replay clips first" }
        }
        assumeTrue(manifest.isFile)
        val rows = mutableListOf("expected,rotation,upright,upright_score,camera,camera_score")
        LandmarkClassifier.create(ModelContract.load(context.assets)).use { classifier ->
            for ((index, line) in manifest.readLines().filter { it.isNotBlank() }.withIndex()) {
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
                val rotation = listOf(90, 270, 180)[index % 3]
                val upright = mutableListOf<HandTracker.Frame>()
                val camera = mutableListOf<HandTracker.Frame>()
                HandTracker.create(context, ModelContract.DETECTOR_ASSET).use { uprightTracker ->
                    HandTracker.create(context, ModelContract.DETECTOR_ASSET).use { cameraTracker ->
                        SequentialClipDecoder.sample(context, uri, durationMs, 32, 640, { true }) { bitmap, ms ->
                            val sideways = Bitmap.createBitmap(
                                bitmap, 0, 0, bitmap.width, bitmap.height,
                                Matrix().apply { postRotate(-rotation.toFloat()) }, false
                            )
                            upright += replay(uprightTracker, bitmap, ms, 0)
                            camera += replay(cameraTracker, sideways, ms, rotation)
                        }
                    }
                }
                assertTrue("Expected enough frames for $clip", upright.size >= 28)
                val result = classify(classifier, upright)
                val rotatedResult = classify(classifier, camera)
                rows += "$expected,$rotation,${result.prediction},${result.confidence}," +
                    "${rotatedResult.prediction},${rotatedResult.confidence}"
            }
        }
        File(directory, "camera-results.csv").writeText(rows.joinToString("\n") + "\n")
        rows.drop(1).forEach { row ->
            val (expected, rotation, upright, _, rotated) = row.split(',')
            assertEquals("Upright replay ($row)", expected, upright)
            assertEquals("Camera replay at $rotation degrees ($row)", upright, rotated)
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

    private fun classify(
        classifier: LandmarkClassifier, frames: List<HandTracker.Frame>
    ): LandmarkClassifier.Result {
        val recording = SignRecorder.Recording(frames, frames.first().capturedAtMs, frames.last().capturedAtMs)
        val features = RecordingFeatures.buildBoth(recording)
        return classifier.predict(features.featuresXY, features.mask, features.featuresXYZ)
    }
}
