package org.echogesture.finalapp

import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomnavigation.BottomNavigationView
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Shell for the four destinations. A single Activity owns the camera, the
 * verified model assets and the four TFLite interpreters, so changing tabs
 * never reloads a model and never restarts the camera.
 *
 * The AI path is the frozen one: [ModelContract] verifies every asset hash and
 * [LandmarkClassifier] runs all nine members through their own interpreters,
 * averages raw logits and applies exactly one softmax.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var contentHost: FrameLayout
    private lateinit var bottomNav: BottomNavigationView

    private lateinit var prefs: AppPreferences
    private lateinit var history: RecognitionHistory
    private lateinit var speaker: Speaker
    private lateinit var diagnostics: DiagnosticsLog
    private lateinit var cameraScreen: CameraScreen
    private lateinit var glossaryScreen: GlossaryScreen
    private lateinit var historyScreen: HistoryScreen
    private lateinit var settingsScreen: SettingsScreen

    private var loaded: LoadedModel? = null
    private var currentTab = R.id.nav_camera

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        contextRef = applicationContext

        contentHost = findViewById(R.id.contentHost)
        bottomNav = findViewById(R.id.bottomNav)

        prefs = AppPreferences(this)
        history = RecognitionHistory(this)
        speaker = Speaker(this)
        diagnostics = DiagnosticsLog(this) { prefs.diagnostics }

        cameraScreen = CameraScreen(this, prefs, history, speaker, diagnostics)
        glossaryScreen = GlossaryScreen(this, history, speaker)
        historyScreen = HistoryScreen(this, history, prefs)
        settingsScreen = SettingsScreen(this, prefs, history, speaker, diagnostics)

        contentHost.addView(
            cameraScreen.root,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )

        bottomNav.setOnItemSelectedListener { item ->
            showTab(item.itemId)
            true
        }
        bottomNav.selectedItemId = R.id.nav_camera
        showTab(R.id.nav_camera)

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (currentTab != R.id.nav_camera) {
                    bottomNav.selectedItemId = R.id.nav_camera
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })

        lifecycleScope.launch { initModels() }
    }

    override fun onResume() {
        super.onResume()
        // A permission granted from system Settings arrives while this activity
        // was stopped, so no ActivityResult is delivered and the Camera tab -
        // whose view tree is retained - would otherwise never notice. The tab
        // re-reads the grant only when it has no live camera.
        if (currentTab == R.id.nav_camera) cameraScreen.onResume()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraScreen.onDestroy()
        speaker.shutdown()
    }

    override fun onPause() {
        cameraScreen.onPause()
        super.onPause()
    }

    // ---- model loading ------------------------------------------------------

    private suspend fun initModels() {
        var tracker: HandTracker? = null
        var classifier: LandmarkClassifier? = null
        var adopted = false
        try {
            val verified = withContext(Dispatchers.IO) {
                val model = ModelContract.load(assets)
                tracker = HandTracker.create(applicationContext, ModelContract.DETECTOR_ASSET)
                classifier = LandmarkClassifier.create(model)
                model
            }
            loaded = verified
            cameraScreen.onModelLoaded(verified, checkNotNull(tracker), checkNotNull(classifier))
            adopted = true
            glossaryScreen.submitLabels(verified.labels)
            settingsScreen.onBind(verified, settingsCallbacks)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (t: Exception) {
            cameraScreen.onModelFailed(t.message ?: t.javaClass.simpleName)
        } finally {
            if (!adopted) withContext(NonCancellable + Dispatchers.IO) {
                try {
                    tracker?.close()
                } finally {
                    classifier?.close()
                }
            }
        }
    }

    private val settingsCallbacks = object : SettingsScreen.Callbacks {
        override fun onPreferencesChanged() = cameraScreen.onPreferencesChanged()
        override fun onHistoryChanged() {
            historyScreen.refresh()
            glossaryScreen.onHistoryChanged()
        }
    }

    // ---- navigation ---------------------------------------------------------

    private fun showTab(itemId: Int) {
        if (currentTab == R.id.nav_camera && itemId != R.id.nav_camera) cameraScreen.onUnbind()
        currentTab = itemId
        contentHost.removeAllViews()
        when (itemId) {
            R.id.nav_camera -> {
                contentHost.addView(cameraScreen.root, match())
                cameraScreen.onBind()
            }
            R.id.nav_glossary -> {
                contentHost.addView(glossaryScreen.root, match())
                loaded?.let { glossaryScreen.submitLabels(it.labels) }
                glossaryScreen.onBind()
            }
            R.id.nav_history -> {
                contentHost.addView(historyScreen.root, match())
                historyScreen.onBind()
            }
            R.id.nav_settings -> {
                contentHost.addView(settingsScreen.root, match())
                settingsScreen.onBind(loaded, settingsCallbacks)
            }
        }
    }

    private fun match() = FrameLayout.LayoutParams(
        ViewGroup.LayoutParams.MATCH_PARENT,
        ViewGroup.LayoutParams.MATCH_PARENT
    )

    /** Exposed for version lookup from Settings without a Context leak. */
    companion object {
        lateinit var contextRef: android.content.Context
    }
}
