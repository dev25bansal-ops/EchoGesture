package org.echogesture.finalapp

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.SystemClock
import android.util.Log
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
import kotlin.math.roundToInt

/**
 * Camera destination: owns the preview, the hand tracker, the capture state
 * machine and the result card.
 *
 * Capture flow
 * ------------
 * READY -> CAPTURING -> PROCESSING -> RESULT
 *
 * CAPTURING clears the tracker's ring and collects a genuinely fresh
 * [HandTracker.CAPTURE_SPAN_MS] window starting from the user's tap, so an
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
    private val speaker: Speaker,
    private val diagnostics: DiagnosticsLog
) {

    /**
     * RECORDING is the explicit mode: the signer started a recording, is
     * performing one sign, and will press Finish. It is a first-class state
     * rather than a flag so the button, the live pill and the timer cannot
     * disagree about whether capture is in progress.
     */
    enum class UiState { INIT, WAITING, RECORDING, PROCESSING, RESULT, NO_HANDS, LOW_FRAMES, ERROR }

    companion object {
        /**
         * Fraction of the capture window that must contain a hand before the
         * window is worth classifying.
         *
         * Measured, not picked by eye: on authorized ASL Citizen clips that
         * the ensemble classifies correctly, the lowest coverage observed was
         * 7/32 frames (22%) and the median was 14/32 (44%). A fraction is used
         * rather than a frame count because the sampled window can be shorter
         * than 32 on a slow device, and 3 frames of 32 (9%) — the previous
         * floor — sits far below any clip that classified correctly.
         *
         * A window below this is reported as an unusable capture rather than
         * scored, so a nearly-empty window cannot produce a confident-looking
         * gloss. It is deliberately below the observed minimum: the first and
         * last frames of a real sign are often empty.
         */
        private const val MIN_HAND_FRACTION = 0.15
    }

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
    private val captureDetailText: TextView = root.findViewById(R.id.captureDetailText)

    private var captureCountdown: Runnable = Runnable { }
    private val confidenceLabel: TextView = root.findViewById(R.id.confidenceLabel)
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

    /** Main-thread handler for the record-mode elapsed-time ticker. */
    private val handler = android.os.Handler(android.os.Looper.getMainLooper())

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

    /** True while the explicit record-mode timer is posted. */
    private var recordingTicker: Runnable? = null

    /**
     * How long the recording has been running, as the UI shows it. Kept in
     * elapsed real time from the same clock the frames are stamped with, so
     * the number on screen and the interval the model receives are the same
     * interval — not two clock domains that happen to look similar.
     */
    private var recordingStartedAtMs = 0L
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
        captureButton.setOnClickListener {
            when (currentState) {
                UiState.RECORDING -> finishRecording()
                UiState.PROCESSING -> Unit // button is disabled, but be explicit
                else -> startRecording()
            }
        }
        helpButton.setOnClickListener { showHelp() }
        retryButton.setOnClickListener {
            haptic(HapticFeedbackConstants.KEYBOARD_TAP)
            startRecording()
        }
        clearButton.setOnClickListener { clearResult() }
        statusAction.setOnClickListener {
            if (currentState == UiState.RECORDING) {
                cancelRecording()
                return@setOnClickListener
            }
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
                modelReady && cameraLive && state != UiState.PROCESSING

            captureButton.text = activity.getString(
                when (state) {
                    UiState.PROCESSING -> R.string.capture_action_processing
                    UiState.RECORDING -> R.string.record_finish
                    else -> R.string.record_start
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
                    guidanceText.visibility =
                        if (prefs.showGuidance) View.VISIBLE else View.GONE
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
                UiState.RECORDING -> {
                    // The live pill carries the running time; guidance sits
                    // under the preview rather than over the signer's hands.
                    hideStatus()
                    setLiveStatus(R.string.record_prompt, R.color.danger)
                    guidanceText.visibility = View.VISIBLE
                    // The primary button is Finish; Cancel is the status
                    // card's secondary action, so a mis-tap costs one press.
                    showStatus(
                        R.string.record_prompt,
                        R.string.record_cancel_hint,
                        icon = R.drawable.ic_hand,
                        tint = R.color.danger,
                        tone = Tone.DANGER,
                        action = true
                    )
                    statusAction.setText(R.string.record_cancel)
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
                UiState.LOW_FRAMES -> {
                    showStatus(
                        R.string.camera_state_low_frames,
                        R.string.low_frames_detail,
                        icon = R.drawable.ic_warning,
                        tint = R.color.warning,
                        tone = Tone.WARNING
                    )
                    setLiveStatus(R.string.camera_state_low_frames, R.color.warning)
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
        captureDetailText.text = ""
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
                // Bound the analysed image so hand scale does not depend on
                // whichever resolution the device's camera happens to default
                // to. Measured on 961 frames of 10 training clips: the detector
                // finds MORE hands at the clip's native resolution (41.4% of
                // frames) than at the 256x256 squash training used (37.6%), so
                // this is a landscape target, not a square one — squashing
                // aspect ratio is what loses detections. Not resizing in
                // HandTracker preserves the detector's own letterboxing.
                .setTargetResolution(android.util.Size(640, 480))
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
        haptic(HapticFeedbackConstants.KEYBOARD_TAP)

        // Classify the window that ENDS at this tap. No waiting: the sign is
        // already over, so anything after the tap is a different gesture.
        val request = t.beginCapture()
        if (request is HandTracker.CaptureRequest.Insufficient) {
            processing = false
            haptic(HapticFeedbackConstants.KEYBOARD_TAP)
            val msg = when (request.reason) {
                // Nothing buffered yet, or the camera has only just started.
                HandTracker.CaptureRequest.Insufficient.Reason.NO_FRAMES,
                HandTracker.CaptureRequest.Insufficient.Reason.TOO_SHORT ->
                    activity.getString(R.string.capture_too_soon)
                // The analyser stopped delivering frames mid-sign.
                HandTracker.CaptureRequest.Insufficient.Reason.STALE ->
                    activity.getString(R.string.capture_stale)
                // A second tap on the same sign would just repeat the answer.
                HandTracker.CaptureRequest.Insufficient.Reason.REPEATED ->
                    activity.getString(R.string.capture_repeated)
            }
            render(UiState.NO_HANDS, detail = msg)
            return
        }
        val capture = (request as HandTracker.CaptureRequest.Ready).capture
        val frames = capture.frames
        activity.runOnUiThread { if (processing) render(UiState.PROCESSING) }
        run {
            inferExecutor.execute {
                try {
                    if (capture.handFrames == 0) {
                        finishProcessing()
                        // A capture with no hand at all is the single most
                        // useful diagnostic line: it separates "the sign was
                        // never seen" from "the model scored it wrong".
                        if (prefs.diagnostics) logDiagnostic(capture, null, 0L)
                        render(UiState.NO_HANDS)
                        return@execute
                    }
                    // A window where the hand is missing for most of it cannot describe a
                    // two-handed dynamic sign; say so rather than producing a
                    // confident-looking guess from a few frames. The onset and
                    // offset frames of a real sign are naturally empty, which is
                    // why this is a fraction and not "the hand in every frame".
                    val coverage = capture.handFrames.toFloat() / capture.frameCount.coerceAtLeast(1)
                    if (coverage < MIN_HAND_FRACTION) {
                        finishProcessing()
                        if (prefs.diagnostics) logDiagnostic(capture, null, 0L)
                        render(UiState.LOW_FRAMES)
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
                    if (prefs.diagnostics) logDiagnostic(capture, result, totalMs)
                    activity.runOnUiThread { renderResult(result, totalMs, capture) }
                } catch (t: Throwable) {
                    finishProcessing()
                    render(UiState.ERROR, detail = activity.getString(R.string.error_inference))
                }
            }
        }
    }

    // ---- explicit record mode ----------------------------------------------

    /**
     * START: collect a fresh temporal sequence from this instant.
     *
     * Nothing from before the press can enter it. The tracker's rolling ring
     * is left alone (quick scan still needs it) and the recorder starts empty,
     * so a previous sign cannot leak into this one.
     */
    private fun startRecording() {
        if (!modelReady || processing) return
        if (!hasCameraPermission()) {
            render(UiState.WAITING, permissionBlocked = true)
            return
        }
        val t = tracker
        if (t == null) {
            render(UiState.ERROR, detail = activity.getString(R.string.error_model_load))
            return
        }
        resultCard.visibility = View.GONE
        recordingStartedAtMs = SystemClock.elapsedRealtime()
        // The recorder uses the same clock instant as the timer below, so the
        // reported duration and the captured interval are the same interval.
        t.startRecording(recordingStartedAtMs)
        haptic(HapticFeedbackConstants.KEYBOARD_TAP)
        render(UiState.RECORDING)
        startRecordingTicker()
    }

    /** FINISH: freeze the interval and classify exactly it. */
    private fun finishRecording() {
        val t = tracker ?: return
        val c = classifier ?: return
        if (processing) return
        stopRecordingTicker()
        processing = true
        val stoppedAt = SystemClock.elapsedRealtime()
        val outcome = t.stopRecording(stoppedAt)
        haptic(HapticFeedbackConstants.KEYBOARD_TAP)

        if (outcome is SignRecorder.Result.Unusable) {
            processing = false
            if (prefs.diagnostics) logRecordingUnusable(outcome)
            render(UiState.NO_HANDS, detail = activity.getString(messageFor(outcome.reason)))
            return
        }
        val recording = (outcome as SignRecorder.Result.Usable).recording
        render(UiState.PROCESSING)
        inferExecutor.execute { runInference(recording, c) }
    }

    private fun cancelRecording() {
        stopRecordingTicker()
        tracker?.cancelRecording()
        processing = false
        haptic(HapticFeedbackConstants.KEYBOARD_TAP)
        Toast.makeText(activity, R.string.record_discarded, Toast.LENGTH_SHORT).show()
        render(UiState.WAITING)
    }

    private fun messageFor(reason: SignRecorder.Rejection): Int = when (reason) {
        SignRecorder.Rejection.TOO_SHORT -> R.string.record_too_short
        SignRecorder.Rejection.NO_FRAMES -> R.string.record_no_frames
        SignRecorder.Rejection.NOT_STARTED -> R.string.record_no_frames
        SignRecorder.Rejection.CAMERA_STALLED -> R.string.record_stalled
        SignRecorder.Rejection.LOW_COVERAGE -> R.string.record_low_coverage
        SignRecorder.Rejection.TOO_LONG -> R.string.record_too_short
    }

    /**
     * Drives the on-screen elapsed time and enforces the upper bound. Posting
     * at 100 ms is enough for a seconds counter and keeps the main thread free.
     */
    private fun startRecordingTicker() {
        stopRecordingTicker()
        val tick = object : Runnable {
            override fun run() {
                if (currentState != UiState.RECORDING) return
                val t = tracker
                val now = SystemClock.elapsedRealtime()
                if (t != null && t.shouldAutoStopRecording(now)) {
                    // Ran past the ceiling: finish with what was captured
                    // rather than growing the buffer without limit.
                    finishRecording()
                    return
                }
                val elapsed = now - recordingStartedAtMs
                liveStatus.text = activity.getString(
                    R.string.record_recording,
                    String.format(java.util.Locale.US, "%.1fs", elapsed / 1000f)
                )
                liveStatus.setTextColor(ContextCompat.getColor(activity, R.color.danger))
                captureButton.text = activity.getString(R.string.record_finish)
                handler.postDelayed(this, 100L)
            }
        }
        recordingTicker = tick
        handler.post(tick)
    }

    private fun stopRecordingTicker() {
        recordingTicker?.let { handler.removeCallbacks(it) }
        recordingTicker = null
    }

    /**
     * The one inference path. Both modes end here: same FeatureTransform, same
     * four pinned interpreters, same mean-logit → single-softmax aggregation.
     * Record mode differs only in which frames it hands over.
     */
    private fun runInference(recording: SignRecorder.Recording, c: LandmarkClassifier) {
        try {
            if (recording.frameCount == 0) {
                processing = false
                render(UiState.NO_HANDS, detail = activity.getString(R.string.record_no_frames))
                return
            }
            val segment = RecordingFeatures.segment(recording)
            val sampled = RecordingFeatures.sample(recording)
            val keptMs = if (sampled.size >= 2)
                sampled.last().capturedAtMs - sampled.first().capturedAtMs else 0L
            val handFrames = sampled.count { it.hands.isNotEmpty() }
            // The card and diagnostics describe the interval actually
            // classified, which is the trimmed one when idle was removed. The
            // recording's full duration is reported separately.
            val capture = HandTracker.Capture(
                sampled,
                keptMs,
                if (keptMs > 0) (sampled.size - 1) * 1000f / keptMs else 0f,
                handFrames
            )
            if (segment.reason == SignSegmenter.Reason.TRIMMED) {
                Log.i("EchoGesture", "segment trimmed to ${segment.endIndex - segment.startIndex}" +
                    "/${recording.frameCount} frames, kept ${keptMs}ms of ${recording.durationMs}ms")
            }
            if (handFrames == 0) {
                processing = false
                if (prefs.diagnostics) logDiagnostic(capture, null, 0L)
                render(UiState.NO_HANDS)
                return
            }
            val coverage = handFrames.toFloat() / sampled.size.coerceAtLeast(1)
            if (coverage < MIN_HAND_FRACTION) {
                processing = false
                if (prefs.diagnostics) logDiagnostic(capture, null, 0L)
                render(UiState.LOW_FRAMES)
                return
            }
            val started = System.nanoTime()
            val features = RecordingFeatures.build(recording)
            val result = c.predict(features)
            val totalMs = (System.nanoTime() - started) / 1_000_000L
            processing = false
            if (prefs.diagnostics) logDiagnostic(capture, result, totalMs)
            activity.runOnUiThread { renderResult(result, totalMs, capture) }
        } catch (t: Throwable) {
            processing = false
            render(UiState.ERROR, detail = activity.getString(R.string.error_inference))
        }
    }

    private fun logRecordingUnusable(outcome: SignRecorder.Result.Unusable) {
        // Diagnostics record the interval and the reason, never pixels. A
        // rejected recording is logged as a capture with no prediction so the
        // rejection rate is measurable alongside accepted captures.
        val r = outcome.recording
        val sampled = r?.let { RecordingFeatures.sample(it) } ?: emptyList()
        diagnostics.record(
            HandTracker.Capture(
                sampled,
                r?.durationMs ?: 0L,
                0f,
                sampled.count { it.hands.isNotEmpty() }
            ),
            null,
            0L
        )
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

    private fun renderResult(
        result: LandmarkClassifier.Result,
        totalMs: Long,
        capture: HandTracker.Capture
    ) {
        val percent = result.confidence * 100f

        // Acceptance is decided by the top-1 minus top-2 margin, not by the
        // score alone. Measured on 82 authorized validation clips: mean margin
        // is 0.378 when the top-1 is correct and 0.064 when it is wrong, and a
        // 0.15 margin cut keeps 61% of captures at 94% accuracy, where the old
        // 0.35 score cut kept 45% at 94.6%. Same accuracy, more captures read
        // out — so the app accepts on margin.
        //
        // Both are still ranking quantities, not probabilities of correctness. The
        // 0.20 score floor is a guard against a degenerate distribution rather
        // than a second filter: on the measured population no capture with a
        // 0.15 margin scored below 0.20, so it does not change which captures
        // are accepted here. It only stops a pathological flat output from
        // being read out as a result.
        val certain = result.margin >= 0.15f && percent >= 20f

        captionText.text = result.prediction
        confidenceBadge.text = activity.getString(
            R.string.result_score,
            "%.1f".format(percent)
        )
        alternativesText.text = result.top5.drop(1).joinToString("  ·  ")

        // Label the reliability, and report the real capture facts next to it.
        // The inference figure alone invited "43 ms" to be read as
        // capture-to-caption, which it never was.
        confidenceLabel.text = activity.getString(
            if (certain) R.string.result_confidence_accepted
            else R.string.result_confidence_uncertain
        )
        // Capture facts and classifier latency are different measurements and
        // are shown as such. The old single line ("43 ms") invited inference
        // timing to be read as capture-to-caption, which it never was.
        captureDetailText.text = activity.getString(
            R.string.result_capture_detail,
            capture.frameCount,
            capture.durationMs,
            capture.fps.roundToInt()
        )
        latencyText.text = activity.getString(
            R.string.result_latency_detail,
            result.inferenceMillis
        )

        val tone = when {
            certain -> R.color.success
            percent >= 20f -> R.color.warning
            else -> R.color.danger
        }
        confidenceBadge.setTextColor(ContextCompat.getColor(activity, tone))
        confidenceBadge.setBackgroundResource(
            when {
                certain -> R.drawable.bg_chip
                percent >= 20f -> R.drawable.bg_card_warning_soft
                else -> R.drawable.bg_card_danger_soft
            }
        )

        render(UiState.RESULT)
        haptic(HapticFeedbackConstants.LONG_PRESS)

        // Still recorded: an uncertain reading is real evidence about what the
        // model saw, and hiding it would lose information the user asked for.
        history.add(result.prediction, result.confidence)
        if (prefs.speech && certain) speakResult(result.prediction)
    }

    /** Runs the diagnostic logger when the user has opted in. Never throws. */
    private fun logDiagnostic(
        capture: HandTracker.Capture,
        result: LandmarkClassifier.Result?,
        totalMs: Long
    ) {
        runCatching { diagnostics.record(capture, result, totalMs) }
            .onFailure { Log.w("CameraScreen", "diagnostic write failed: ${it.javaClass.simpleName}") }
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