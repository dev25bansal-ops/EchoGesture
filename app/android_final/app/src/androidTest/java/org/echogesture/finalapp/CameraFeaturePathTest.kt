package org.echogesture.finalapp

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Camera-path and temporal-layout checks that do not need a real hand in frame.
 *
 * [EnsembleInferenceTest] proves the classifier's numbers. Here we prove two
 * things the directive requires of the *input* side:
 *
 *  - the analysed image's rotation is reported by CameraX and accepted by the
 *    shipped [HandTracker], which is what MainActivity feeds to MediaPipe
 *    exactly once through ImageProcessingOptions (never baked into the bitmap);
 *  - a frame with no hand keeps its position in the 24-frame sequence instead
 *    of being dropped, so a sign is not compressed in time.
 *
 * The camera test runs the real CameraX -> ImageAnalysis -> HandTracker path
 * against the live device sensor (the virtual one on an emulator). Its
 * simulated camera may report rotation 0; a locked-portrait phone reports 90.
 * The assertion is therefore "a valid quarter-turn, accepted by the tracker",
 * and the on-device value is recorded by the device procedure in
 * docs/demo/ANDROID_DEVICE_TEST.md.
 */
@RunWith(AndroidJUnit4::class)
class CameraFeaturePathTest {

    private val instrumentation
        get() = InstrumentationRegistry.getInstrumentation()

    /** Minimal lifecycle so CameraX can bind outside an Activity. */
    private class TestOwner : LifecycleOwner {
        private val registry = LifecycleRegistry(this)
        override val lifecycle: Lifecycle get() = registry
        fun start() {
            registry.handleLifecycleEvent(Lifecycle.Event.ON_CREATE)
            registry.handleLifecycleEvent(Lifecycle.Event.ON_START)
            registry.handleLifecycleEvent(Lifecycle.Event.ON_RESUME)
        }
        fun stop() {
            registry.handleLifecycleEvent(Lifecycle.Event.ON_PAUSE)
            registry.handleLifecycleEvent(Lifecycle.Event.ON_STOP)
            registry.handleLifecycleEvent(Lifecycle.Event.ON_DESTROY)
        }
    }

    @Test
    fun cameraReportsValidRotationAndTrackerAcceptsFrames() {
        val camera = instrumentation.targetContext.getSystemService(
            android.hardware.camera2.CameraManager::class.java
        )
        if (camera.cameraIdList.isEmpty()) {
            return // no camera on this device: nothing to assert, do not fake a pass
        }
        // Instrumentation runs in the app's process; grant CAMERA here because
        // `connectedDebugAndroidTest` cannot answer a runtime-permission dialog.
        instrumentation.uiAutomation.grantRuntimePermission(
            instrumentation.targetContext.packageName,
            android.Manifest.permission.CAMERA
        )

        val rotations = ArrayList<Int>()
        val firstFrame = CountDownLatch(1)
        val analysed = AtomicInteger(0)
        val executor = Executors.newSingleThreadExecutor()
        val tracker = HandTracker.create(instrumentation.targetContext, ModelContract.DETECTOR_ASSET)
        val owner = TestOwner()

        ProcessCameraProvider.getInstance(instrumentation.targetContext).get(30, TimeUnit.SECONDS)
            .let { provider ->
                val analysis = ImageAnalysis.Builder().build().also { a ->
                    a.setAnalyzer(executor) { proxy: ImageProxy ->
                        val rotation = proxy.imageInfo.rotationDegrees
                        val accepted = tracker.analyse(proxy)
                        if (accepted) {
                            analysed.incrementAndGet()
                            synchronized(rotations) { rotations.add(rotation) }
                            if (analysed.get() >= 3) firstFrame.countDown()
                        }
                    }
                }
                // The emulator's virtual sensor may not advertise a back lens,
                // so ask CameraX for whatever exists instead of hardcoding a
                // facing: this test is about rotation reporting and tracker
                // acceptance, not lens selection (which MainActivity owns).
                val selector = CameraSelector.Builder()
                    .requireLensFacing(
                        if (provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)) {
                            androidx.camera.core.CameraSelector.LENS_FACING_BACK
                        } else {
                            androidx.camera.core.CameraSelector.LENS_FACING_FRONT
                        }
                    )
                    .build()
                // CameraX requires binding/unbinding on the main thread.
                instrumentation.runOnMainSync {
                    provider.unbindAll()
                    owner.start()
                    provider.bindToLifecycle(owner, selector, analysis)
                }
                try {
                    assertTrue(
                        "no camera frame was analysed within 20 s",
                        firstFrame.await(20, TimeUnit.SECONDS)
                    )
                } finally {
                    instrumentation.runOnMainSync {
                        provider.unbindAll()
                        owner.stop()
                    }
                    tracker.close()
                    executor.shutdown()
                }
            }

        assertTrue("expected analysed frames", analysed.get() >= 3)
        synchronized(rotations) {
            rotations.forEach {
                assertTrue(
                    "CameraX reported rotation $it, not a quarter-turn (0/90/180/270)",
                    it in setOf(0, 90, 180, 270)
                )
            }
        }
        // Record the observed value for the device report without failing on it:
        // an emulator sensor is legitimately different from a phone.
        android.util.Log.i("CameraFeaturePathTest", "observed rotations: $rotations")
    }

    /**
     * The app's own feature assembly: a 32-frame window in which the hand is
     * present only in the middle. The source frame count passed to
     * [FeatureTransform] is the whole window, not the hand count, so the
     * handless frames keep their temporal position.
     */
    @Test
    fun missingHandFramesKeepTheirPosition() {
        val frames = 32
        val landmarks = FloatArray(frames * 2 * 21 * 3)
        val present = FloatArray(frames * 2)
        // Right hand (slot 1) with a non-degenerate wrist->middleMCP scale,
        // present only in source frames 10..19.
        for (f in 10 until 20) {
            val base = f * 2 * 21 * 3 + 1 * 21 * 3
            landmarks[base + 0 * 3] = 0.1f
            landmarks[base + 9 * 3] = 0.2f
            present[f * 2 + 1] = 1f
        }

        // Same call MainActivity.toFeatureArrays -> buildFeatures performs.
        val feats = FeatureTransform.buildFeatures(landmarks, present, frames, 24)
        assertEquals(24 * 255, feats.size)

        // Output frame 0 maps to source frame 0.0 and frame 23 to source 31.0.
        // Both are handless, so their normalised block must stay zeros instead
        // of borrowing a hand from a later/earlier frame.
        assertTrue("frame 0 is handless and must stay zeros", (0 until 84).all { feats[it] == 0f })
        assertTrue(
            "frame 23 is handless and must stay zeros",
            (23 * 255 until 23 * 255 + 84).all { feats[it] == 0f }
        )
        val carrying = (0 until 24).count { o ->
            (o * 255 until o * 255 + 84).any { feats[it] != 0f }
        }
        assertTrue("hand frames must survive, got $carrying", carrying >= 5)
        assertTrue("hand frames must not fill the whole window, got $carrying", carrying <= 19)
    }
}