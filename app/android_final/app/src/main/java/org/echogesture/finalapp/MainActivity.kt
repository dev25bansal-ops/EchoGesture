package org.echogesture.finalapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/**
 * Fully offline camera -> MediaPipe HandLandmarker -> 24-frame / 255-feature ->
 * TFLite classifier caption flow.
 *
 * - No INTERNET or network-state permission is declared; frames and results
 *   never leave the device.
 * - 32 analysed frames are retained; Capture resamples to 24 and classifies on
 *   background executors. Inference uses the pinned FP16 model only.
 * - HUD states: waiting / processing / caption / no-hands / error, plus a
 *   permanent OFFLINE badge, model identity, and timings.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var captionText: android.widget.TextView
    private lateinit var confidenceText: android.widget.TextView
    private lateinit var alternativesText: android.widget.TextView
    private lateinit var latencyText: android.widget.TextView
    private lateinit var statusTitle: android.widget.TextView
    private lateinit var statusDetail: android.widget.TextView
    private lateinit var offlineIndicator: android.widget.TextView
    private lateinit var modelIdentity: android.widget.TextView
    private lateinit var frameStatus: android.widget.TextView
    private lateinit var captureButton: android.widget.Button
    private lateinit var retryButton: android.widget.Button
    private lateinit var permissionButton: android.widget.Button

    private var cameraProvider: ProcessCameraProvider? = null
    private var tracker: HandTracker? = null
    private var classifier: LandmarkClassifier? = null
    private var contract: LoadedModel? = null

    private val trackerExecutor = Executors.newSingleThreadExecutor()
    private val inferExecutor = Executors.newSingleThreadExecutor()

    @Volatile private var processing = false
    @Volatile private var modelReady = false

    private enum class State { INIT, WAITING, PROCESSING, CAPTION, NO_HANDS, ERROR }
    private var state = State.INIT

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() else render(State.WAITING, permissionBlocked = true) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        modelIdentity.text = "Loading model…"
        render(State.INIT)

        lifecycleScope.launch(Dispatchers.IO) { initModelAssets() }
        if (hasCameraPermission()) startCamera() else permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    override fun onDestroy() {
        super.onDestroy()
        tracker?.close()
        classifier?.close()
        tracker = null
        classifier = null
        trackerExecutor.shutdown()
        inferExecutor.shutdown()
    }

    // ---- UI -----------------------------------------------------------------

    private fun bindViews() {
        previewView = findViewById(R.id.preview)
        captionText = findViewById(R.id.captionText)
        confidenceText = findViewById(R.id.confidenceText)
        alternativesText = findViewById(R.id.alternativesText)
        latencyText = findViewById(R.id.latencyText)
        statusTitle = findViewById(R.id.statusTitle)
        statusDetail = findViewById(R.id.statusDetail)
        offlineIndicator = findViewById(R.id.offlineIndicator)
        modelIdentity = findViewById(R.id.modelIdentity)
        frameStatus = findViewById(R.id.frameStatus)
        captureButton = findViewById(R.id.captureButton)
        retryButton = findViewById(R.id.retryButton)
        permissionButton = findViewById(R.id.permissionButton)

        captureButton.setOnClickListener { captureAndPredict() }
        retryButton.setOnClickListener {
            tracker?.clear()
            clearCaption()
            render(State.WAITING)
        }
        permissionButton.setOnClickListener {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun render(state: State, detail: String? = null, permissionBlocked: Boolean = false) {
        runOnUiThread {
            this.state = state
            offlineIndicator.text = getString(R.string.offline_indicator)
            captureButton.isEnabled = modelReady && state != State.PROCESSING
            permissionButton.visibility =
                if (permissionBlocked) android.view.View.VISIBLE else android.view.View.GONE

            when (state) {
                State.INIT -> {
                    statusTitle.visibility = android.view.View.VISIBLE
                    statusDetail.visibility = android.view.View.VISIBLE
                    retryButton.visibility = android.view.View.GONE
                    statusTitle.text = "Initializing…"
                    statusDetail.text = detail ?: "Loading the on-device detector and classifier."
                }
                State.WAITING -> {
                    statusTitle.visibility = android.view.View.VISIBLE
                    statusDetail.visibility = android.view.View.VISIBLE
                    retryButton.visibility = android.view.View.GONE
                    statusTitle.text = getString(
                        if (permissionBlocked) R.string.permission_title else R.string.waiting_title
                    )
                    statusDetail.text = getString(
                        if (permissionBlocked) R.string.permission_detail else R.string.waiting_detail
                    )
                }
                State.PROCESSING -> {
                    statusTitle.visibility = android.view.View.VISIBLE
                    statusDetail.visibility = android.view.View.VISIBLE
                    retryButton.visibility = android.view.View.GONE
                    statusTitle.text = getString(R.string.processing_title)
                    statusDetail.text = getString(R.string.processing_detail)
                }
                State.CAPTION -> {
                    statusTitle.visibility = android.view.View.GONE
                    statusDetail.visibility = android.view.View.GONE
                    retryButton.visibility = android.view.View.VISIBLE
                }
                State.NO_HANDS -> {
                    statusTitle.visibility = android.view.View.VISIBLE
                    statusDetail.visibility = android.view.View.VISIBLE
                    retryButton.visibility = android.view.View.VISIBLE
                    statusTitle.text = getString(R.string.no_hands_title)
                    statusDetail.text = getString(R.string.no_hands_detail)
                }
                State.ERROR -> {
                    statusTitle.visibility = android.view.View.VISIBLE
                    statusDetail.visibility = android.view.View.VISIBLE
                    retryButton.visibility = android.view.View.VISIBLE
                    statusTitle.text = getString(R.string.error_title)
                    statusDetail.text = detail ?: "Something went wrong."
                }
            }
        }
    }

    private fun clearCaption() {
        captionText.text = ""
        confidenceText.text = ""
        alternativesText.text = ""
        latencyText.text = ""
    }

    private fun updateFrameStatus(frames: Int, handFrames: Int) = runOnUiThread {
        frameStatus.text = getString(R.string.frame_status, frames.coerceAtMost(32), handFrames)
    }

    // ---- Model and camera ---------------------------------------------------

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private suspend fun initModelAssets() = withContext(Dispatchers.IO) {
        try {
            val loaded = ModelContract.load(assets)
            val newTracker = HandTracker.create(this@MainActivity, ModelContract.DETECTOR_ASSET)
            // Throws unless every listed member loads with its own interpreter.
            val newClassifier = LandmarkClassifier.create(loaded)

            tracker?.close()
            classifier?.close()
            tracker = newTracker
            classifier = newClassifier
            contract = loaded

            withContext(Dispatchers.Main) {
                modelIdentity.text = getString(
                    R.string.model_identity,
                    loaded.identity.displayName,
                    loaded.identity.shortSha,
                    newClassifier.memberCount
                )
                modelReady = true
                if (state == State.INIT) render(State.WAITING)
            }
        } catch (t: Throwable) {
            withContext(Dispatchers.Main) {
                render(State.ERROR, detail = "Model load failed: ${t.message ?: t.javaClass.simpleName}")
            }
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                cameraProvider = provider

                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { a ->
                        a.setAnalyzer(trackerExecutor) { imageProxy ->
                            val t = tracker
                            if (t != null) {
                                if (t.analyse(imageProxy)) {
                                    val (frames, handFrames) = t.status()
                                    updateFrameStatus(frames, handFrames)
                                }
                            } else {
                                imageProxy.close()
                            }
                        }
                    }

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
                render(State.WAITING)
            } catch (t: Throwable) {
                render(State.ERROR, detail = "Camera failed to start: ${t.message ?: t.javaClass.simpleName}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ---- Capture and inference ----------------------------------------------

    private fun captureAndPredict() {
        if (!modelReady || processing) return
        val t = tracker
        val c = classifier
        val m = contract
        if (t == null || c == null || m == null) {
            render(State.ERROR, detail = "Model not ready yet.")
            return
        }

        val frames = t.snapshot().filter { it.hands.isNotEmpty() }
        if (frames.isEmpty()) {
            render(State.NO_HANDS)
            return
        }

        processing = true
        render(State.PROCESSING)
        val started = System.nanoTime()

        inferExecutor.execute {
            try {
                val (landmarks, present, srcFrames) = toFeatureArrays(frames)
                val feats = FeatureTransform.buildFeatures(landmarks, present, srcFrames, 24)
                val result = c.predict(feats)
                val totalMs = (System.nanoTime() - started) / 1_000_000L
                processing = false
                renderCaption(result, totalMs)
            } catch (t: Throwable) {
                processing = false
                runOnUiThread {
                    render(State.ERROR, detail = "Inference failed: ${t.message ?: t.javaClass.simpleName}")
                }
            }
        }
    }

    /** Collapse the ring into flattened (T,2,21,3) and (T,2) arrays. */
    private fun toFeatureArrays(frames: List<HandTracker.Frame>): Triple<FloatArray, FloatArray, Int> {
        val n = frames.size
        val landmarks = FloatArray(n * 2 * 21 * 3)
        val present = FloatArray(n * 2)
        frames.forEachIndexed { fi, fr ->
            for (hand in fr.hands) {
                present[fi * 2 + hand.slot] = 1f
                System.arraycopy(hand.xyz, 0, landmarks, (fi * 2 + hand.slot) * 21 * 3, 21 * 3)
            }
        }
        return Triple(landmarks, present, n)
    }

    private fun renderCaption(result: LandmarkClassifier.Result, totalMs: Long) = runOnUiThread {
        state = State.CAPTION
        captionText.text = result.prediction
        confidenceText.text = getString(
            R.string.caption_confidence,
            "%.1f".format(result.confidence * 100f)
        )
        alternativesText.text = getString(
            R.string.top_alternatives,
            result.top5.drop(1).joinToString(", ")
        )
        latencyText.text = getString(R.string.latency_line, result.inferenceMillis, totalMs)
        render(State.CAPTION)
    }
}