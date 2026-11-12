package org.echogesture.finalapp

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.view.HapticFeedbackConstants
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.DecelerateInterpolator
import android.widget.Button
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.TorchState
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.material.bottomsheet.BottomSheetDialog
import java.util.concurrent.Executors

/**
 * Camera destination: owns the preview, the hand tracker, the capture state
 * machine and the result card.
 *
 * Capture flow
 * ------------
 * READY -> CAPTURING -> PROCESSING -> RESULT
 *
 * CAPTURING clears the tracker's ring and collects a genuinely fresh
 * [HandTracker.RING_CAPACITY]-frame window starting from the user's tap, so an
 * old window and a new sign can never be mixed. When the window is full the
 * existing 32->24 resample and four-member ensemble run unchanged on the
 * inference executor.
 *
 * The AI core is untouched: the same FeatureTransform, the same four pinned
 * interpreters, the same mean-logit -> single-softmax aggregation.
 */
class CameraScreen(
    private val activity: MainActivity,
    private val prefs: AppPreferences,
    private val history: RecognitionHistory,
    private val speaker: Speaker
) {

    enum class UiState { INIT, WAITING, CAPTURING, PROCESSING, RESULT, NO_HANDS, ERROR }

    val root: View = LayoutInflater.from(activity).inflate(R.layout.screen_camera, null, false)

    private val previewView: PreviewView = root.findViewById(R.id.preview)
    private val previewContainer: LinearLayout? = null
    private val previewFrame: View = root.findViewById(R.id.previewContainer)
    private val frameGuide: org.echogesture.finalapp.ui.FrameGuideView =
        root.findViewById(R.id.frameGuide)
    private val guidanceText: TextView = root.findViewById(R.id.guidanceText)
    private val liveStatus: TextView = root.findViewById(R.id.liveStatus)
    private val captureButton: Button = root.findViewById(R.id.captureButton)

    private val resultCard: View = root.findViewById(R.id.resultCard)
    private val captionText: TextView = root.findViewById(R.id.captionText)
    private val confidenceBadge: TextView = root.findViewById(R.id.confidenceBadge)
    private val alternativesText: TextView = root.findViewById(R.id.alternativesText)
    private val latencyText: TextView = root.findViewById(R.id.latencyText)
    private val copyButton: Button = root.findViewById(R.id.copyButton)
    private val speakButton: Button = root.findViewById(R.id.speakButton)
    private val retryButton: Button = root.findViewById(R.id.retryButton)
    private val clearButton: Button = root.findViewById(R.id.clearButton)

    private val statusCard: View = root.findViewById(R.id.statusCard)
    private val statusIcon: ImageView = root.findViewById(R.id.statusIcon)
    private val statusTitle: TextView = root.findViewById(R.id.statusTitle)
    private val statusDetail: TextView = root.findViewById(R.id.statusDetail)
    private val statusAction: Button = root.findViewById(R.id.statusAction)

    private val permissionCard: View = root.findViewById(R.id.permissionCard)
    private val permissionButton: Button = root.findViewById(R.id.permissionButton)
    private val modelIdentity: TextView = root.findViewById(R.id.modelIdentity)
    private val helpButton: Button = root.findViewById(R.id.helpButton)
    private val flipButton: ImageButton = root.findViewById(R.id.flipButton)
    private val torchButton: ImageButton = root.findViewById(R.id.torchButton)

    private val trackerExecutor = Executors.newSingleThreadExecutor()
    private val inferExecutor = Executors.newSingleThreadExecutor()

    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var preview: Preview? = null

    @Volatile private var tracker: HandTracker? = null
    @Volatile private var classifier: LandmarkClassifier? = null
    @Volatile private var contractContract: LoadedModel? = null
    @Volatile private var modelReady = false
    @Volatile private var processing = false
    @Volatile private var trackerFailureReported = false
    @Volatile private var boundSelector: CameraSelector? = null

    private var currentState = UiState.INIT
    private var lastFrames = 0
    private var lastHandFrames = 0
    private var bound = false

    private val permissionLauncher = activity.registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            permissionCard.visibility = View.GONE
            startCamera(prefs.cameraChoice)
        } else {
            render(UiState.WAITING, permissionBlocked = true)
        }
    }

    // ---- lifecycle ----------------------------------------------------------

    fun onBind() {
        if (bound) return
        bound = true
        wireControls()
        frameGuide.attachTo(previewView)
        applyPreferences()
        modelIdentity.text = activity.getString(R.string.camera_state_preparing)
        render(UiState.INIT)
        speaker.initialise()

        if (hasCameraPermission()) startCamera(prefs.cameraChoice)
        else permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    fun onUnbind() {
        if (!bound) return
        bound = false
        runCatching { tracker?.cancelCapture() }
        runCatching { cameraProvider?.unbindAll() }
        camera = null
        imageAnalysis = null
        preview = null
    }

    /** Called by the Activity once the assets are verified and interpreters exist. */
    fun onModelLoaded(loaded: LoadedModel, newTracker: HandTracker, newClassifier: LandmarkClassifier) {
        tracker?.close()
        classifier?.close()
        tracker = newTracker
        classifier = newClassifier
        contractContract = loaded
        modelReady = true
        modelIdentity.text = activity.getString(
            R.string.settings_model_ensemble
        ) + " · " + activity.getString(
            R.string.settings_model_precision
        ) + " · SHA-256 " + loaded.identity.shortSha + "…"
        if (currentState == UiState.INIT) render(UiState.WAITING)
        // The analyser may be bound before the models finish loading.
        rebindAnalyserIfNeeded()
    }

    fun onModelFailed(message: String) {
        modelReady = false
        render(UiState.ERROR, detail = activity.getString(R.string.error_model_load))
        modelIdentity.text = message
    }

    fun onDestroy() {
        tracker?.close()
        classifier?.close()
        tracker = null
        classifier = null
        trackerExecutor.shutdownNow()
        inferExecutor.shutdownNow()
    }

    fun currentCaption(): String? =
        if (currentState == UiState.RESULT) captionText.text.toString() else null

    // ---- wiring -------------------------------------------------------------

    private fun wireControls() {
        captureButton.setOnClickListener { beginCapture() }
        helpButton.setOnClickListener { showHelp() }
        retryButton.setOnClickListener {
            haptic(HapticFeedbackConstants.KEYBOARD_TAP)
            beginCapture()
        }
        clearButton.setOnClickListener { clearResult() }
        statusAction.setOnClickListener {
            when (currentState) {
                UiState.ERROR -> {
                    trackerFailureReported = false
                    startCamera(prefs.cameraChoice)
                    render(UiState.WAITING)
                }
                else -> {
                    trackerFailureReported = false
                    render(UiState.WAITING)
                }
            }
        }
        permissionButton.setOnClickListener {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
        flipButton.setOnClickListener {
            haptic(HapticFeedbackConstants.KEYBOARD_TAP)
            switchCamera()
        }
        torchButton.setOnClickListener { toggleTorch() }
        copyButton.setOnClickListener {
            val text = captionText.text?.toString().orEmpty()
            if (text.isNotEmpty()) {
                copyToClipboard(text)
                Toast.makeText(activity, R.string.action_copied, Toast.LENGTH_SHORT).show()
                haptic(HapticFeedbackConstants.KEYBOARD_TAP)
            }
        }
        speakButton.setOnClickListener {
            val text = captionText.text?.toString().orEmpty()
            if (text.isNotEmpty()) speakResult(text)
        }
    }

    private fun applyPreferences() {
        val sizeSp = prefs.captionSp()
        captionText.setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, sizeSp)
        frameGuide.visibility = if (prefs.showGuidance) View.VISIBLE else View.GONE
        guidanceText.visibility = if (prefs.showGuidance) View.VISIBLE else View.GONE
    }

    fun onPreferencesChanged() = applyPreferences()

    // ---- rendering ----------------------------------------------------------

    private fun render(
        state: UiState,
        detail: String? = null,
        permissionBlocked: Boolean = false
    ) {
        activity.runOnUiThread {
            currentState = state
            permissionCard.visibility = if (permissionBlocked) View.VISIBLE else View.GONE

            val cameraLive = hasCameraPermission() && camera != null
            captureButton.isEnabled =
                modelReady && cameraLive && state != UiState.CAPTURING && state != UiState.PROCESSING

            captureButton.text = activity.getString(
                when (state) {
                    UiState.CAPTURING -> R.string.capture_action_capturing
                    UiState.PROCESSING -> R.string.capture_action_processing
                    else -> R.string.capture_action
                }
            )

            when (state) {
                UiState.INIT -> {
                    showStatus(
                        R.string.camera_state_preparing,
                        R.string.camera_state_preparing,
                        icon = R.drawable.ic_hand,
                        tint = R.color.text_secondary
                    )
                    setLiveStatus(R.string.camera_state_preparing, R.color.text_on_dark)
                }
                UiState.WAITING -> {
                    if (permissionBlocked) {
                        hideStatus()
                        setLiveStatus(R.string.permission_title, R.color.warning)
                    } else {
                        val handsSeen = lastHandFrames.coerceAtLeast(0)
                        hideStatus()
                        setLiveStatus(
                            if (handsSeen > 0) R.string.camera_state_detected
                            else R.string.camera_state_searching,
                            R.color.text_on_dark
                        )
                    }
                }
                UiState.CAPTURING -> {
                    hideStatus()
                    setLiveStatus(R.string.camera_state_capturing, R.color.text_on_dark)
                }
                UiState.PROCESSING -> {
                    hideStatus()
                    setLiveStatus(R.string.camera_state_processing, R.color.text_on_dark)
                }
                UiState.RESULT -> {
                    hideStatus()
                    setLiveStatus(R.string.result_label, R.color.success)
                    resultCard.visibility = View.VISIBLE
                    revealResult()
                }
                UiState.NO_HANDS -> {
                    showStatus(
                        R.string.camera_state_no_hands,
                        R.string.no_hands_detail_for_state,
                        icon = R.drawable.ic_warning,
                        tint = R.color.warning,
                        tone = Tone.WARNING
                    )
                    setLiveStatus(R.string.camera_state_no_hands, R.color.warning)
                }
                UiState.ERROR -> {
                    showStatus(
                        R.string.error_title,
                        detail?.let { text ->
                            // Keep diagnostic wording out of the primary card.
                            R.string.error_generic
                        } ?: R.string.error_generic,
                        icon = R.drawable.ic_warning,
                        tint = R.color.danger,
                        tone = Tone.DANGER,
                        action = true
                    )
                    if (detail != null) statusDetail.text = detail
                    setLiveStatus(R.string.error_title, R.color.danger)
                }
            }
        }
    }

    /** The three tones a status card can take. Colour is never the only cue. */
    private enum class Tone { NEUTRAL, WARNING, DANGER }

    private fun showStatus(
        titleRes: Int,
        detailRes: Int,
        icon: Int,
        @androidx.annotation.ColorRes tint: Int,
        tone: Tone = Tone.NEUTRAL,
        action: Boolean = false
    ) {
        statusCard.visibility = View.VISIBLE
        resultCard.visibility = View.GONE
        statusTitle.setText(titleRes)
        statusDetail.setText(detailRes)
        statusIcon.setImageResource(icon)
        statusIcon.imageTintList =
            android.content.res.ColorStateList.valueOf(ContextCompat.getColor(activity, tint))
        statusCard.setBackgroundResource(
            when (tone) {
                Tone.DANGER -> R.drawable.bg_card_danger_soft
                Tone.WARNING -> R.drawable.bg_card_warning_soft
                Tone.NEUTRAL -> R.drawable.bg_card
            }
        )
        statusAction.visibility = if (action) View.VISIBLE else View.GONE
    }

    private fun hideStatus() {
        if (currentState != UiState.RESULT) resultCard.visibility = View.GONE
        statusCard.visibility = View.GONE
    }

    private fun setLiveStatus(textRes: Int, @androidx.annotation.ColorRes tint: Int) {
        liveStatus.setText(textRes)
        liveStatus.setTextColor(ContextCompat.getColor(activity, tint))
    }

    private fun revealResult() {
        // 180 ms fade + small rise. Restrained, and skipped when the system
        // animator scale is off so reduced-motion users get no motion.
        if (android.provider.Settings.Global.getFloat(
                activity.contentResolver,
                android.provider.Settings.Global.ANIMATOR_DURATION_SCALE,
                1f
            ) == 0f
        ) {
            resultCard.alpha = 1f
            resultCard.translationY = 0f
            return
        }
        resultCard.alpha = 0f
        resultCard.translationY = 12f * activity.resources.displayMetrics.density
        resultCard.animate()
            .alpha(1f)
            .translationY(0f)
            .setDuration(180L)
            .setInterpolator(DecelerateInterpolator())
            .start()
    }

    private fun clearResult() {
        resultCard.visibility = View.GONE
        captionText.text = ""
        confidenceBadge.text = ""
        alternativesText.text = ""
        latencyText.text = ""
        render(UiState.WAITING)
        haptic(HapticFeedbackConstants.KEYBOARD_TAP)
    }

    // ---- camera -------------------------------------------------------------

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun selectorFor(choice: AppPreferences.CameraChoice): CameraSelector {
        val provider = cameraProvider
        val rearAvailable = provider?.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA) == true
        val frontAvailable = provider?.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA) == true
        return when (choice) {
            AppPreferences.CameraChoice.REAR -> CameraSelector.DEFAULT_BACK_CAMERA
            AppPreferences.CameraChoice.FRONT -> CameraSelector.DEFAULT_FRONT_CAMERA
            // AUTO keeps the organizer's exocentric third-person scenario:
            // someone films the signer, which is the rear camera.
            AppPreferences.CameraChoice.AUTO ->
                if (rearAvailable || !frontAvailable) CameraSelector.DEFAULT_BACK_CAMERA
                else CameraSelector.DEFAULT_FRONT_CAMERA
        }
    }

    private fun startCamera(choice: AppPreferences.CameraChoice) {
        val future = ProcessCameraProvider.getInstance(activity)
        future.addListener({
            try {
                val provider = future.get()
                cameraProvider = provider
                bindUseCases(selectorFor(choice))
            } catch (t: Throwable) {
                render(UiState.ERROR, detail = activity.getString(R.string.error_camera_start))
            }
        }, ContextCompat.getMainExecutor(activity))
    }

    private fun analyserCallback(imageProxy: androidx.camera.core.ImageProxy) {
        val t = tracker
        if (t == null) {
            imageProxy.close()
            return
        }
        if (t.analyse(imageProxy)) {
            trackerFailureReported = false
            val (frames, handFrames) = t.status()
            lastFrames = frames
            lastHandFrames = handFrames
            activity.runOnUiThread {
                if (currentState == UiState.WAITING) {
                    setLiveStatus(
                        if (handFrames > 0) R.string.camera_state_detected
                        else R.string.camera_state_searching,
                        R.color.text_on_dark
                    )
                }
            }
        } else if (t.consecutiveFailures() >= HandTracker.FAILURE_THRESHOLD) {
            reportPersistentTrackerFailure()
        }
    }

    private fun bindUseCases(selector: CameraSelector) {
        val provider = cameraProvider ?: return
        try {
            provider.unbindAll()

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { a -> a.setAnalyzer(trackerExecutor) { analyserCallback(it) } }
            imageAnalysis = analysis

            val newPreview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            preview = newPreview

            camera = provider.bindToLifecycle(activity, selector, newPreview, analysis)
            boundSelector = selector
            updateTorchVisibility()
            render(if (modelReady) UiState.WAITING else UiState.INIT)
        } catch (t: Throwable) {
            render(UiState.ERROR, detail = activity.getString(R.string.error_camera_start))
        }
    }

    private fun rebindAnalyserIfNeeded() {
        // The tracker is created after the analyser in a cold start; nothing to
        // do if the camera is already bound, because the analyser reads the
        // tracker field on each frame rather than capturing it.
    }

    private fun switchCamera() {
        if (!hasCameraPermission()) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
            return
        }
        val current = boundSelector
        val next = if (current == CameraSelector.DEFAULT_BACK_CAMERA ||
            (current == null && prefs.cameraChoice != AppPreferences.CameraChoice.FRONT)
        ) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }
        // Persist the explicit choice so the next launch matches what the user
        // just picked instead of silently reverting to AUTO.
        prefs.cameraChoice = if (next == CameraSelector.DEFAULT_BACK_CAMERA) {
            AppPreferences.CameraChoice.REAR
        } else {
            AppPreferences.CameraChoice.FRONT
        }
        // A camera rebind changes sensor orientation; start from a clean ring so
        // no frame from the previous camera reaches the model.
        tracker?.clear()
        render(UiState.WAITING)
        bindUseCases(next)
    }

    private fun updateTorchVisibility() {
        val hasFlash = camera?.cameraInfo?.hasFlashUnit() == true
        torchButton.visibility = if (hasFlash) View.VISIBLE else View.GONE
        torchButton.contentDescription = activity.getString(R.string.torch_toggle_desc)
    }

    private fun toggleTorch() {
        val cam = camera ?: return
        if (cam.cameraInfo.hasFlashUnit() != true) return
        val on = cam.cameraInfo.torchState.value == TorchState.ON
        cam.cameraControl.enableTorch(!on)
        haptic(HapticFeedbackConstants.KEYBOARD_TAP)
    }

    private fun reportPersistentTrackerFailure() {
        if (trackerFailureReported) return
        trackerFailureReported = true
        render(UiState.ERROR, detail = activity.getString(R.string.error_camera_start))
    }

    // ---- capture ------------------------------------------------------------

    private fun beginCapture() {
        if (!modelReady || processing) return
        if (!hasCameraPermission()) {
            render(UiState.WAITING, permissionBlocked = true)
            return
        }
        val t = tracker
        val c = classifier
        val m = contractContract
        if (t == null || c == null || m == null) {
            render(UiState.ERROR, detail = activity.getString(R.string.error_model_load))
            return
        }
        if (processing) return
        processing = true
        resultCard.visibility = View.GONE
        render(UiState.CAPTURING)
        haptic(HapticFeedbackConstants.KEYBOARD_TAP)

        // Collect a fresh window starting now, not whatever was already in the
        // ring. The callback fires on the analysis thread once the window is
        // full, then inference moves to its own executor.
        t.beginCapture {
            if (!processing) return@beginCapture
            val frames = t.captureSnapshot()
            activity.runOnUiThread { if (processing) render(UiState.PROCESSING) }
            inferExecutor.execute {
                try {
                    if (frames.none { it.hands.isNotEmpty() }) {
                        finishProcessing()
                        render(UiState.NO_HANDS)
                        return@execute
                    }
                    val started = System.nanoTime()
                    val (landmarks, present, srcFrames) = toFeatureArrays(frames)
                    // Same 32->24 resample and feature contract as the Python
                    // reference (STRICT_RESAMPLE=False).
                    val feats = FeatureTransform.buildFeatures(landmarks, present, srcFrames, 24)
                    val result = c.predict(feats)
                    val totalMs = (System.nanoTime() - started) / 1_000_000L
                    finishProcessing()
                    activity.runOnUiThread { renderResult(result, totalMs) }
                } catch (t: Throwable) {
                    finishProcessing()
                    render(UiState.ERROR, detail = activity.getString(R.string.error_inference))
                }
            }
        }
    }

    private fun finishProcessing() {
        processing = false
    }

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

    private fun renderResult(result: LandmarkClassifier.Result, totalMs: Long) {
        captionText.text = result.prediction
        val percent = result.confidence * 100f
        confidenceBadge.text = activity.getString(
            R.string.result_score,
            "%.1f".format(percent)
        )
        alternativesText.text = result.top5.drop(1).joinToString("  ·  ")
        latencyText.text = activity.getString(R.string.result_processing_time, result.inferenceMillis)

        val tone = when {
            percent >= 60f -> R.color.success
            percent >= 30f -> R.color.warning
            else -> R.color.danger
        }
        confidenceBadge.setTextColor(ContextCompat.getColor(activity, tone))
        confidenceBadge.setBackgroundResource(
            when {
                percent >= 60f -> R.drawable.bg_chip
                percent >= 30f -> R.drawable.bg_card_warning_soft
                else -> R.drawable.bg_card_danger_soft
            }
        )

        render(UiState.RESULT)
        haptic(HapticFeedbackConstants.LONG_PRESS)

        history.add(result.prediction, result.confidence)
        if (prefs.speech) speakResult(result.prediction)
    }

    private fun speakResult(text: String) {
        val spoken = speaker.speak(text)
        if (!spoken) {
            Toast.makeText(activity, R.string.tts_unavailable, Toast.LENGTH_SHORT).show()
        }
    }

    private fun copyToClipboard(text: String) {
        val cm = activity.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText(activity.getString(R.string.app_name), text))
    }

    private fun haptic(feedback: Int) {
        if (!prefs.haptics) return
        // Respect the system touch-feedback setting first; then give VIBRATOR
        // callers a real short pulse on devices without a haptic motor.
        if (!android.provider.Settings.System.getInt(
                activity.contentResolver,
                android.provider.Settings.System.HAPTIC_FEEDBACK_ENABLED,
                1
            ).let { it != 0 }
        ) return
        root.performHapticFeedback(feedback)
    }

    /** Shows the short, accurate usage guide. Content is the real flow, not a mock-up. */
    fun showHelp() {
        val dialog = BottomSheetDialog(activity)
        val view = LayoutInflater.from(activity).inflate(R.layout.sheet_help, null, false)
        view.findViewById<Button>(R.id.helpClose).setOnClickListener { dialog.dismiss() }
        dialog.setContentView(view)
        dialog.show()
    }
}