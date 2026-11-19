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

    /**
     * Finish a recording automatically when the sign stops moving, instead of
     * waiting for a second tap.
     *
     * Default ON because the tap is a measured source of error: on the Vivo
     * V2404 a capture ran 1540 ms and held 4 frames, and another held the hand
     * for only 11 of 32 frames, because the press landed at the wrong moment.
     * The signer cannot see the analyser's real rate, so they cannot time a
     * press to it. The motion signal can.
     *
     * Off is offered because auto-finish has a cost a manual press does not:
     * a signer who pauses mid-sign is cut short. Anyone who prefers explicit
     * control can turn it off and keep the old behaviour exactly.
     */
    var autoFinish: Boolean
        get() = prefs.getBoolean(KEY_AUTO_FINISH, true)
        set(v) = prefs.edit().putBoolean(KEY_AUTO_FINISH, v).apply()

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
        const val KEY_AUTO_FINISH = "auto_finish"
        const val KEY_CAPTION_SIZE = "caption_size"
    }
}