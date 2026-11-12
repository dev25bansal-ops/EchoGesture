package org.echogesture.finalapp

import android.content.Context
import android.content.SharedPreferences

/**
 * Local-only settings. SharedPreferences is used deliberately: the values are
 * a handful of primitives, they must survive process death, and they must never
 * leave the device. There is no network path here and no account.
 */
class AppPreferences(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_FILE, Context.MODE_PRIVATE)

    enum class CameraChoice { AUTO, REAR, FRONT }


    enum class CaptionSize { SMALL, MEDIUM, LARGE }

    var cameraChoice: CameraChoice
        get() = enumValue(KEY_CAMERA, CameraChoice.AUTO)
        set(v) = prefs.edit().putString(KEY_CAMERA, v.name).apply()

    var showGuidance: Boolean
        get() = prefs.getBoolean(KEY_GUIDANCE, true)
        set(v) = prefs.edit().putBoolean(KEY_GUIDANCE, v).apply()

    var haptics: Boolean
        get() = prefs.getBoolean(KEY_HAPTICS, true)
        set(v) = prefs.edit().putBoolean(KEY_HAPTICS, v).apply()

    var speech: Boolean
        get() = prefs.getBoolean(KEY_SPEECH, false)
        set(v) = prefs.edit().putBoolean(KEY_SPEECH, v).apply()

    /**
     * Off by default. History is an in-memory record of what was recognised,
     * which is still personal data about a signer; keeping it only for the
     * session is the conservative default and saving it is an explicit choice.
     */
    var saveHistory: Boolean
        get() = prefs.getBoolean(KEY_SAVE_HISTORY, false)
        set(v) = prefs.edit().putBoolean(KEY_SAVE_HISTORY, v).apply()

    /**
     * Off by default and only ever switched on for a consenting test session.
     * Records capture counts, timing and presence flags — never images,
     * landmarks or identifiers. See [DiagnosticsLog].
     */
    var diagnostics: Boolean
        get() = prefs.getBoolean(KEY_DIAGNOSTICS, false)
        set(v) = prefs.edit().putBoolean(KEY_DIAGNOSTICS, v).apply()

    var captionSize: CaptionSize
        get() = enumValue(KEY_CAPTION_SIZE, CaptionSize.MEDIUM)
        set(v) = prefs.edit().putString(KEY_CAPTION_SIZE, v.name).apply()

    fun captionSp(): Float = when (captionSize) {
        CaptionSize.SMALL -> 28f
        CaptionSize.MEDIUM -> 36f
        CaptionSize.LARGE -> 46f
    }

    private inline fun <reified T : Enum<T>> enumValue(key: String, fallback: T): T {
        val stored = prefs.getString(key, null) ?: return fallback
        return runCatching { enumValueOf<T>(stored) }.getOrDefault(fallback)
    }

    companion object {
        /** Preference backing-store name, shared with [RecognitionHistory]. */
        const val PREFS_FILE = "echogesture_prefs"

        const val KEY_CAMERA = "camera_choice"
        const val KEY_GUIDANCE = "show_guidance"
        const val KEY_HAPTICS = "haptics"
        const val KEY_SPEECH = "speech"
        const val KEY_SAVE_HISTORY = "save_history"
        const val KEY_DIAGNOSTICS = "diagnostics"
        const val KEY_CAPTION_SIZE = "caption_size"
    }
}