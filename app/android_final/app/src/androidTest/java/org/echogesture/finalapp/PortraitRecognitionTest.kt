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
                val portrait = track(file, index)
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

    // Freeze the previous release so the comparison remains reproducible as defaults evolve.
    private val v58 = AutoSegmenter.Config(
        acquireMs = 300, poseHoldMs = 450, preRollMs = 550, quietHoldMs = 800, handLossGraceMs = 900
    )

    @Test
    fun automaticResponseOnLabeledPortraitClips() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.filesDir, "file-corpus")
        val manifest = File(directory, "response.tsv")
        val required = InstrumentationRegistry.getArguments().getString("requireResponse") == "true"
        if (required) check(manifest.isFile) { "Stage the automatic response replay first" }
        assumeTrue(manifest.isFile)
        val clips = manifest.readLines().filter { it.isNotBlank() }
        if (required) check(clips.size == 120)
        val rows = mutableListOf("clip,expected,version,label,score,accepted,trigger_ms,reason,compute_ms")
        LandmarkClassifier.create(ModelContract.load(context.assets)).use { classifier ->
            for ((index, line) in clips.withIndex()) {
                val (clip, expected) = line.split('\t')
                require(File(clip).name == clip)
                val input = stream(track(File(directory, clip), index))
                for (baseline in listOf(true, false)) {
                    val version = if (baseline) "5.8" else "5.9"
                    val detector = AutoSegmenter(if (baseline) v58 else AutoSegmenter.Config())
                    detector.arm(0)
                    var trigger = 0L
                    val outcome = input.firstNotNullOfOrNull { frame ->
                        detector.offer(frame)?.also { trigger = frame.capturedAtMs }
                    }
                    val start = android.os.SystemClock.elapsedRealtime()
                    var result = ",0,false"
                    if (outcome?.isUsable == true) {
                        val recording = SignRecorder.Recording(outcome.frames, outcome.startedAtMs, outcome.endedAtMs)
                        val segment = if (baseline) RecordingFeatures.segment(recording) else RecordingFeatures.autoSegment(recording)
                        val samples = RecordingFeatures.sample(recording, segment)
                        if (samples.count { it.hands.isNotEmpty() } >= samples.size * .15) {
                            val inputs = RecordingFeatures.buildBoth(samples)
                            val p = classifier.predict(inputs.featuresXY, inputs.mask, inputs.featuresXYZ)
                            val accepted = (baseline || ResultPolicy.hasSustainedTracking(samples)) &&
                                ResultPolicy.isAccepted(p.margin, p.confidence * 100f)
                            result = "${p.prediction},${p.confidence},$accepted"
                        }
                    }
                    val compute = android.os.SystemClock.elapsedRealtime() - start
                    rows += "$clip,$expected,$version,$result,$trigger,${outcome?.reason ?: "NO_SIGN"},$compute"
                }
            }
        }
        File(directory, "response-results.csv").writeText(rows.joinToString("\n", postfix = "\n"))
        if (required) {
            val pairs = rows.drop(1).map { it.split(',') }.chunked(2)
            assertTrue("Correct first choices must improve", pairs.count { it[1][1] == it[1][3] } >
                pairs.count { it[0][1] == it[0][3] })
            val savings = pairs.filter { it.all { row -> row[3].isNotEmpty() } }
                .map { it[0][6].toLong() - it[1][6].toLong() }.sorted()
            assertTrue("Median automatic completion should be faster", savings[savings.size / 2] > 0)
        }
    }

    @Test
    fun compareWideAndCloseFramingForDistantSigners() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.filesDir, "file-corpus")
        val manifest = File(directory, "manifest.tsv")
        val required = InstrumentationRegistry.getArguments().getString("requireRearFraming") == "true"
        if (required) check(manifest.isFile)
        assumeTrue(manifest.isFile)
        val rows = mutableListOf("clip,expected,zoom,prediction,score,accepted")
        LandmarkClassifier.create(ModelContract.load(context.assets)).use { classifier ->
            for ((index, line) in manifest.readLines().filter { it.isNotBlank() }.withIndex()) {
                val (clip, expected) = line.split('\t')
                require(File(clip).name == clip)
                for (zoom in listOf(1f, CameraScreen.REAR_CLOSE_ZOOM)) {
                    val frames = track(File(directory, clip), index, (240 * zoom).toInt())
                    val detector = AutoSegmenter()
                    detector.arm(0)
                    val outcome = stream(frames).firstNotNullOfOrNull(detector::offer)
                    val prediction = if (outcome?.isUsable == true) {
                        classify(classifier, outcome.frames, automatic = true)
                    } else ",0,false"
                    rows += "$clip,$expected,$zoom,$prediction"
                }
            }
        }
        File(directory, "rear-framing-results.csv").writeText(rows.joinToString("\n", postfix = "\n"))
    }

    @Test
    fun compareCameraConfirmations() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.filesDir, "file-corpus")
        val manifest = File(directory, "camera-verification.tsv")
        val required = InstrumentationRegistry.getArguments().getString("requireCameraVerification") == "true"
        if (required) check(manifest.isFile)
        assumeTrue(manifest.isFile)
        val lines = manifest.readLines()
        if (required) check(lines.size == 80)
        File(directory, "camera-verification.csv").printWriter().use { out ->
            out.println("clip,expected,width,baseline,baseline_score,baseline_accepted,baseline_ms,current,current_score,current_accepted,current_ms")
            LandmarkClassifier.create(ModelContract.load(context.assets)).use { classifier ->
                for ((index, line) in lines.withIndex()) {
                    val (clip, label) = line.split('\t')
                    require(File(clip).name == clip)
                    val width = if (index % 4 < 2) 480 else 240
                    val detector = AutoSegmenter()
                    detector.arm(0)
                    val outcome = stream(track(File(directory, clip), index, width)).firstNotNullOfOrNull(detector::offer)
                    val predictions = if (outcome?.isUsable == true) {
                        val rec = SignRecorder.Recording(outcome.frames, outcome.startedAtMs, outcome.endedAtMs)
                        val samples = RecordingFeatures.sample(rec, RecordingFeatures.autoSegment(rec))
                        if (samples.count { it.hands.isNotEmpty() } >= samples.size * .15) {
                            listOf(false, true).joinToString(",") { retry ->
                                val start = System.nanoTime()
                                val scored = scoreRecording(classifier, rec, samples, retry)
                                val elapsedMs = (System.nanoTime() - start) / 1_000_000L
                                val p = scored.result
                                val accepted = ResultPolicy.hasSustainedTracking(scored.frames) &&
                                    (scored.corroborated || ResultPolicy.isAccepted(p.margin, p.confidence * 100f))
                                "${p.prediction},${p.confidence},$accepted,$elapsedMs"
                            }
                        } else ",0,false,0,,0,false,0"
                    } else ",0,false,0,,0,false,0"
                    out.println("$clip,$label,$width,$predictions")
                    out.flush()
                }
            }
        }
    }

    @Test
    fun measureModerateCameraMatches() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(context.filesDir, "file-corpus")
        val manifest = File(dir, "camera64.tsv")
        val required = InstrumentationRegistry.getArguments().getString("requireModerateReplay") == "true"
        if (required) check(manifest.isFile)
        assumeTrue(manifest.isFile)
        val clips = manifest.readLines()
        if (required) check(clips.size == 120)
        File(dir, "moderate.csv").printWriter().use { out ->
            out.println("clip,label,width,prediction,score,margin,votes,v63,v64,ms,frozen_extra")
            LandmarkClassifier.create(ModelContract.load(context.assets)).use { classifier ->
                for ((index, line) in clips.withIndex()) {
                    val (name, label) = line.split('\t')
                    require(File(name).name == name)
                    val width = if (index % 4 < 2) 480 else 240
                    val frames = track(File(dir, name), index, width)
                    val detector = AutoSegmenter()
                    detector.arm(0)
                    val outcome = stream(frames).firstNotNullOfOrNull(detector::offer)
                    var data = ",0,0,0,false,false,0,false"
                    if (outcome?.isUsable == true) {
                        val rec = SignRecorder.Recording(outcome.frames, outcome.startedAtMs, outcome.endedAtMs)
                        val samples = RecordingFeatures.sample(rec, RecordingFeatures.autoSegment(rec))
                        if (samples.count { it.hands.isNotEmpty() } >= samples.size * .15) {
                            val start = System.nanoTime()
                            val scored = scoreRecording(classifier, rec, samples, true)
                            val elapsed = (System.nanoTime() - start) / 1_000_000
                            val p = scored.result
                            val old = ResultPolicy.hasSustainedTracking(scored.frames) &&
                                ResultPolicy.isAccepted(p.margin, p.confidence * 100f)
                            val held = samples.first { it.hands.isNotEmpty() }
                            val frozen = rec.copy(frames = rec.frames.map {
                                if (it.hands.isEmpty()) it else held.copy(capturedAtMs = it.capturedAtMs)
                            })
                            val heldSamples = RecordingFeatures.sample(frozen, RecordingFeatures.autoSegment(frozen))
                            val frozenExtra = scoreRecording(classifier, frozen, heldSamples, true).corroborated
                            assertTrue("A stationary hand must not acquire a new confirmation", !frozenExtra)
                            data = "${p.prediction},${p.confidence},${p.margin},${p.agreeingMembers},$old,${old || scored.corroborated},$elapsed,$frozenExtra"
                        }
                    }
                    out.println("$name,$label,$width,$data")
                    out.flush()
                }
            }
        }
    }

    private fun track(file: File, index: Int, frameWidth: Int = 480): List<HandTracker.Frame> {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val rotation = if (index % 2 == 0) 90 else 270
        val uri = Uri.fromFile(file)
        val retriever = MediaMetadataRetriever()
        val durationMs = try {
            retriever.setDataSource(file.path)
            checkNotNull(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)).toLong()
        } finally {
            retriever.release()
        }
        val portrait = mutableListOf<HandTracker.Frame>()
        HandTracker.create(context, ModelContract.DETECTOR_ASSET).use { tracker ->
            SequentialClipDecoder.sample(context, uri, durationMs, 32, 640, { true }) { bitmap, ms ->
                val tall = Bitmap.createBitmap(480, 640, Bitmap.Config.ARGB_8888)
                val h = frameWidth * bitmap.height / bitmap.width
                val left = (480 - frameWidth) / 2
                val top = (640 - h) / 2
                try {
                    Canvas(tall).apply {
                        drawColor(Color.DKGRAY)
                        drawBitmap(bitmap, null, Rect(left, top, left + frameWidth, top + h), null)
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
        assertTrue("Expected enough frames for ${file.name}", portrait.size >= 28)
        return portrait
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
        val s = AutoSegmenter(v58)
        s.arm(0)
        val result = stream(frames).firstNotNullOfOrNull(s::offer)
        return if (result?.isUsable == true) classify(classifier, result.frames) else ",0,false"
    }

    private fun stream(frames: List<HandTracker.Frame>): List<HandTracker.Frame> {
        val lead = 600L
        val inputs = frames.map { it.copy(capturedAtMs = it.capturedAtMs + lead) }
        val tail = inputs.last().capturedAtMs
        return listOf(150L, 300L, 450L).map { HandTracker.Frame(emptyList(), it) } +
            inputs + (100L..1200L step 100L).map { HandTracker.Frame(emptyList(), tail + it) }
    }

    private fun classify(
        classifier: LandmarkClassifier, frames: List<HandTracker.Frame>, automatic: Boolean = false
    ): String {
        if (frames.none { it.hands.isNotEmpty() }) return ",0,false"
        val recording = SignRecorder.Recording(frames, frames.first().capturedAtMs, frames.last().capturedAtMs)
        val samples = if (automatic) RecordingFeatures.sample(recording, RecordingFeatures.autoSegment(recording))
            else RecordingFeatures.sample(recording)
        val features = RecordingFeatures.buildBoth(samples)
        val p = classifier.predict(features.featuresXY, features.mask, features.featuresXYZ)
        val accepted = (!automatic || ResultPolicy.hasSustainedTracking(samples)) &&
            ResultPolicy.isAccepted(p.margin, p.confidence * 100f)
        return "${p.prediction},${p.confidence},$accepted"
    }
}
