package org.echogesture.finalapp

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Wraps Android's local TextToSpeech engine. Nothing is synthesised in the
 * cloud: [TextToSpeech] uses whichever voice packs are installed on the device,
 * and the app holds no network permission, so an unavailable offline voice is
 * reported rather than silently substituted.
 */
class Speaker(context: Context) {

    interface StatusListener {
        fun onAvailabilityChanged(available: Boolean)
    }

    private val appContext = context.applicationContext
    private var tts: TextToSpeech? = null
    private val available = AtomicBoolean(false)
    private val initialised = AtomicBoolean(false)
    private val mainHandler = Handler(Looper.getMainLooper())
    var listener: StatusListener? = null

    fun initialise() {
        if (!initialised.compareAndSet(false, true)) return
        tts = TextToSpeech(appContext) { status ->
            mainHandler.post {
                val engine = tts ?: return@post
                val offlineVoices = if (status == TextToSpeech.SUCCESS) {
                    engine.voices.orEmpty().filter {
                        it.locale.language == Locale.ENGLISH.language &&
                            !it.isNetworkConnectionRequired &&
                            TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED !in it.features.orEmpty()
                    }
                } else {
                    emptyList()
                }
                val voice = offlineVoices.firstOrNull { it.locale == Locale.US }
                    ?: offlineVoices.firstOrNull()
                available.set(voice != null && engine.setVoice(voice) == TextToSpeech.SUCCESS)
                listener?.onAvailabilityChanged(available.get())
            }
        }
    }

    fun isAvailable(): Boolean = available.get()

    /**
     * Speaks [text] only after an installed local voice has been selected.
     */
    fun speak(text: String): Boolean {
        val engine = tts ?: run {
            initialise()
            return false
        }
        if (!available.get() || engine.voice?.isNetworkConnectionRequired != false) return false
        return speakNow(text, engine)
    }

    private fun speakNow(text: String, engine: TextToSpeech? = tts): Boolean {
        val e = engine ?: return false
        val result = e.speak(text, TextToSpeech.QUEUE_FLUSH, null, UTTERANCE_ID)
        return result == TextToSpeech.SUCCESS
    }

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        available.set(false)
    }

    private companion object {
        const val UTTERANCE_ID = "echogesture_result"
    }
}
