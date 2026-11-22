package org.echogesture.finalapp

import android.view.LayoutInflater
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.switchmaterial.SwitchMaterial

/**
 * Settings that genuinely change app behaviour. Every control here writes to
 * [AppPreferences] or [RecognitionHistory] and takes effect immediately; there
 * are no decorative toggles.
 */
class SettingsScreen(
    private val activity: MainActivity,
    private val prefs: AppPreferences,
    private val history: RecognitionHistory,
    private val diagnostics: DiagnosticsLog
) {

    val root: View = LayoutInflater.from(activity).inflate(R.layout.screen_settings, null, false)

    interface Callbacks {
        fun onPreferencesChanged()
        fun onHistoryChanged()
    }

    private lateinit var callbacks: Callbacks

    private val cameraSpinner: Spinner = root.findViewById(R.id.preferredCamera)
    private val switchGuidance: SwitchMaterial = root.findViewById(R.id.switchGuidance)
    private val switchHaptics: SwitchMaterial = root.findViewById(R.id.switchHaptics)
    private val switchSpeech: SwitchMaterial = root.findViewById(R.id.switchSpeech)
    private val switchSaveHistory: SwitchMaterial = root.findViewById(R.id.switchSaveHistory)
    private val switchDiagnostics: SwitchMaterial = root.findViewById(R.id.switchDiagnostics)
    private val captionGroup: RadioGroup = root.findViewById(R.id.captionSizeGroup)
    private val captionSmall: RadioButton = root.findViewById(R.id.captionSmall)
    private val captionMedium: RadioButton = root.findViewById(R.id.captionMedium)
    private val captionLarge: RadioButton = root.findViewById(R.id.captionLarge)
    private val clearHistoryButton: Button = root.findViewById(R.id.clearHistoryButton)
    private val exportDiagnosticsButton: Button = root.findViewById(R.id.exportDiagnosticsButton)
    private val modelHash: TextView = root.findViewById(R.id.modelHash)
    private val modelMembers: TextView = root.findViewById(R.id.modelMembers)
    private val aboutVersion: TextView = root.findViewById(R.id.aboutVersion)

    private var bound = false

    fun onBind(loaded: LoadedModel?, callbacks: Callbacks) {
        this.callbacks = callbacks
        if (!bound) {
            bound = true
            wireControls()
        }
        bindValues()
        bindModelInfo(loaded)
    }

    private fun wireControls() {
        val choices = arrayOf(
            activity.getString(R.string.camera_rear),
            activity.getString(R.string.camera_front),
            activity.getString(R.string.camera_auto)
        )
        cameraSpinner.adapter = ArrayAdapter(
            activity,
            android.R.layout.simple_spinner_dropdown_item,
            choices
        )
        cameraSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            var initialised = false
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                if (!initialised) { initialised = true; return }
                prefs.cameraChoice = when (position) {
                    0 -> AppPreferences.CameraChoice.REAR
                    1 -> AppPreferences.CameraChoice.FRONT
                    else -> AppPreferences.CameraChoice.AUTO
                }
                callbacks.onPreferencesChanged()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        switchGuidance.setOnCheckedChangeListener { _, checked ->
            prefs.showGuidance = checked
            callbacks.onPreferencesChanged()
        }
        switchHaptics.setOnCheckedChangeListener { _, checked ->
            prefs.haptics = checked
            callbacks.onPreferencesChanged()
        }
        switchSpeech.setOnCheckedChangeListener { _, checked ->
            prefs.speech = checked
            callbacks.onPreferencesChanged()
        }
        switchSaveHistory.setOnCheckedChangeListener { _, checked ->
            prefs.saveHistory = checked
            history.onPersistenceChanged(checked)
            callbacks.onHistoryChanged()
        }
        switchDiagnostics.setOnCheckedChangeListener { _, checked ->
            prefs.diagnostics = checked
            // Turning it off deletes what was recorded: the point of the switch
            // is that a test session's record does not outlive the session.
            if (!checked) diagnostics.clear()
        }
        captionGroup.setOnCheckedChangeListener { _, checkedId ->
            prefs.captionSize = when (checkedId) {
                R.id.captionSmall -> AppPreferences.CaptionSize.SMALL
                R.id.captionLarge -> AppPreferences.CaptionSize.LARGE
                else -> AppPreferences.CaptionSize.MEDIUM
            }
            callbacks.onPreferencesChanged()
        }
        clearHistoryButton.setOnClickListener {
            val count = history.size()
            if (count == 0) {
                Toast.makeText(activity, R.string.history_empty_title, Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            MaterialAlertDialogBuilder(activity)
                .setTitle(R.string.settings_clear_history)
                .setMessage(activity.getString(R.string.history_clear_confirm, count))
                .setNegativeButton(R.string.action_cancel, null)
                .setPositiveButton(R.string.history_clear) { _, _ ->
                    history.clear()
                    callbacks.onHistoryChanged()
                    Toast.makeText(activity, R.string.history_cleared, Toast.LENGTH_SHORT).show()
                }
                .show()
        }
        exportDiagnosticsButton.setOnClickListener { exportDiagnostics() }
    }

    /**
     * Hands the diagnostics record to whatever the user picks in the system
     * share sheet.
     *
     * Nothing leaves the device until they choose a target, and the app has no
     * INTERNET permission, so the only routes are the ones the user selects.
     * An empty record is reported rather than sharing a blank file.
     */
    private fun exportDiagnostics() {
        val file = diagnostics.exportCopy(
            java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.US)
                .format(java.util.Date())
        )
        if (file == null) {
            Toast.makeText(activity, R.string.settings_export_empty, Toast.LENGTH_LONG).show()
            return
        }
        runCatching {
            val uri = androidx.core.content.FileProvider.getUriForFile(
                activity, "${activity.packageName}.fileprovider", file
            )
            val send = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(android.content.Intent.EXTRA_STREAM, uri)
                putExtra(android.content.Intent.EXTRA_SUBJECT, "EchoGesture diagnostics")
                addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            activity.startActivity(
                android.content.Intent.createChooser(
                    send, activity.getString(R.string.settings_export_chooser)
                )
            )
        }.onFailure {
            Toast.makeText(activity, R.string.settings_export_empty, Toast.LENGTH_LONG).show()
        }
    }

    private fun bindValues() {
        cameraSpinner.setSelection(
            when (prefs.cameraChoice) {
                AppPreferences.CameraChoice.REAR -> 0
                AppPreferences.CameraChoice.FRONT -> 1
                AppPreferences.CameraChoice.AUTO -> 2
            },
            false
        )
        switchGuidance.isChecked = prefs.showGuidance
        switchHaptics.isChecked = prefs.haptics
        switchSpeech.isChecked = prefs.speech
        switchSaveHistory.isChecked = prefs.saveHistory
        switchDiagnostics.isChecked = prefs.diagnostics
        when (prefs.captionSize) {
            AppPreferences.CaptionSize.SMALL -> captionSmall.isChecked = true
            AppPreferences.CaptionSize.MEDIUM -> captionMedium.isChecked = true
            AppPreferences.CaptionSize.LARGE -> captionLarge.isChecked = true
        }
        aboutVersion.text = activity.getString(R.string.settings_about_version, BuildVersion.name)
    }

    private fun bindModelInfo(loaded: LoadedModel?) {
        if (loaded == null) {
            modelHash.text = "—"
            modelMembers.text = ""
            return
        }
        modelHash.text = loaded.identity.combinedSha256
        modelMembers.text = loaded.identity.memberNames.joinToString("\n")
    }

    /** Real versionName from the installed package, not a hardcoded string. */
    private object BuildVersion {
        val name: String
            get() = runCatching {
                val app = org.echogesture.finalapp.MainActivity.contextRef
                app.packageManager.getPackageInfo(app.packageName, 0).versionName ?: "1.0"
            }.getOrDefault("1.0")
    }
}