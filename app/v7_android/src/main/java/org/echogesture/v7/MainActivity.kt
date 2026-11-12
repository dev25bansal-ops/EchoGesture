package org.echogesture.v7

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.HandLandmarker
import com.google.mediapipe.tasks.vision.HandLandmarker.HandLandmarkerOptions
import com.google.mediapipe.tasks.vision.HandLandmarker.RunningMode
import com.google.mediapipe.tasks.vision.HandLandmarker.PolygonLandmark
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.Image
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.util.Size
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Minimal offline vertical slice:
 *   CameraX preview (off-device pixels never leave the phone; NO network
 *   permission is declared) -> MediaPipe HandLandmarker (VIDEO mode, 2 hands)
 *   -> FeatureTransform (Kotlin port of src/features.py) -> TFLite classifier.
 *
 * Frames are sampled to 24 evenly across a 32-frame read window, exactly as
 * src/landmark_frontend.read_frames / src/main.py do on the host. Inference
 * runs once per "capture" tap; preview keeps running. No video is saved.
 */
class MainActivity : AppCompatActivity() {

    private var landmarker: HandLandmarker? = null
    private var classifier: Classifier? = null
    private var labels: List<String> = emptyList()
    private val ring = ArrayDeque<Frame>(32)
    private var frameTs = 0L
    private var cameraExecutor = Executors.newSingleThreadExecutor()

    data class Frame(val bitmap: Bitmap, val handed: List<Pair<Int, FloatArray>>)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initModelAssetsAsync()
        // Wire CameraX here in the app's UI layer (PreviewView + ImageCapture
        // "Capture" button). Kept view-agnostic: startCamera() below only
        // needs a PreviewView surface from the layout.
        // No network permission is requested in AndroidManifest.xml.
    }

    private fun initModelAssetsAsync() = lifecycleScope.launch(Dispatchers.IO) {
        // Copy .tflite / .task assets to filesDir (MediaPipe requires a file path)
        // and read labels.json. Skipped here to keep the slice free of
        // asset-copy boilerplate; the exact asset bytes and SHA-256 pins are
        // in ../../../../evaluation/device/v7_android/ and model_contract.json.
        // Expected landmarker options (mirror landmark_frontend.py):
        //   num_hands=2, min_detection/existing/tracking = 0.3
    }

    private fun buildLandmarker(): HandLandmarker {
        val opts = HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.3f)
            .setMinHandPresenceConfidence(0.3f)
            .setMinTrackingConfidence(0.3f)
            .build()
        return HandLandmarker.createFromOptions(this, opts)
    }

    // Called from the CameraX ImageAnalysis analyser (off the main thread).
    private fun onFrame(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        imageProxy.close()
        if (landmarker == null) landmarker = buildLandmarker()
        frameTs += 33
        val mpImage = Image.fromBitmap(bitmap)
        val result = landmarker!!.detectForVideo(mpImage, frameTs)
        val handed = result.handLandmarks().mapIndexed { i, pts ->
            val cat = result.handedness()[i][0].categoryName() // "Left"/"Right"
            val slot = if (cat == "Left") 0 else 1
            val arr = FloatArray(21 * 3)
            pts.forEachIndexed { j, p -> arr[j * 3] = p.x(); arr[j * 3 + 1] = p.y(); arr[j * 3 + 2] = p.z() }
            slot to arr
        }
        synchronized(ring) {
            ring.addLast(Frame(bitmap, handed))
            while (ring.size > 32) ring.removeFirst()
        }
    }

    // "Capture" tap: resample the ring to 24 frames, build features, classify.
    fun captureAndPredict(): Map<String, Any> {
        val frames: List<Frame> = synchronized(ring) { ring.toList() }
        require(frames.isNotEmpty()) { "no frames captured yet" }
        val t = frames.size
        val landmarks = FloatArray(t * 2 * 21 * 3)
        val present = FloatArray(t * 2)
        frames.forEachIndexed { fi, fr ->
            fr.handed.forEach { (slot, arr) ->
                System.arraycopy(arr, 0, landmarks, (fi * 2 + slot) * 21 * 3, 21 * 3)
                present[fi * 2 + slot] = 1f
            }
        }
        val feats = FeatureTransform.buildFeatures(landmarks, present, t, 24)
        val res = classifier!!.predict(feats, labels)
        return mapOf("prediction" to res.prediction, "top5" to res.top5, "confidence" to res.confidence)
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        // YUV_420_888 -> RGB bitmap. CameraX can also hand us Bitmap directly
        // via ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST; a production build uses
        // ML Kit's YuvImage path. Stub: convert via android.graphics.YuvImage.
        val yuv = planes[0].buffer; val u = planes[1].buffer; val v = planes[2].buffer
        val ySize = yuv.remaining(); val uSize = u.remaining(); val vSize = v.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yuv.get(nv21, 0, ySize); v.get(nv21, ySize, vSize); u.get(nv21, ySize + vSize, uSize)
        val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
        val out = android.graphics.Bitmap.createBitmap(width, height, android.graphics.Bitmap.Config.ARGB_8888)
        val data = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, data)
        return android.graphics.BitmapFactory.decodeByteArray(data.toByteArray(), 0, data.size())
    }

    override fun onDestroy() {
        landmarker?.close(); classifier?.close()
        landmarker = null; classifier = null
        cameraExecutor.shutdown(); cameraExecutor.awaitTermination(1, TimeUnit.SECONDS)
        super.onDestroy()
    }
}
