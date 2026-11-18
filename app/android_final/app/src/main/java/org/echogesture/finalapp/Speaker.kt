package org.echogesture.finalapp

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
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
    private var pending: String? = null
    var listener: StatusListener? = null

    fun initialise() {
        if (!initialised.compareAndSet(false, true)) return
        tts = TextToSpeech(appContext) { status ->
            val engine = tts
            if (status == TextToSpeech.SUCCESS && engine != null) {
                val lang = engine.setLanguage(Locale.US)
                val usable = lang != TextToSpeech.LANG_MISSING_DATA &&
                    lang != TextToSpeech.LANG_NOT_SUPPORTED
                available.set(usable)
                engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onDone(utteranceId: String?) {}
                    @Deprecated("Deprecated in Java")
                    override fun onError(utteranceId: String?) {}
                })
            } else {
                available.set(false)
                // Allow a later retry to re-initialise. Without this, a TTS
                // service that was momentarily unavailable at launch left
                // `initialised` true for the whole session and every subsequent
                // initialise() no-oped, so "Speak results" could never recover
                // and permanently reported "No offline voice available".
                runCatching { tts?.shutdown() }
                tts = null
                initialised.set(false)
            }
            listener?.onAvailabilityChanged(available.get())
            pending?.let { text ->
                pending = null
                if (available.get()) speakNow(text)
            }
        }
    }

    fun isAvailable(): Boolean = available.get()

    /**
     * Speaks [text]. If the engine is still initialising the request is queued
     * and spoken once it becomes available. Returns false when no offline voice
     * exists so the caller can show an explicit unavailable state.
     */
    fun speak(text: String): Boolean {
        val engine = tts ?: run {
            pending = text
            initialise()
            return false
        }
        if (!available.get()) {
            // Still initialising, or no voice at all. Queue only while the init
            // callback has not landed yet; once it has and reported a failure,
            // `tts` is null again (see initialise's failure branch) so this
            // branch means "in flight". Queueing here is what the doc above
            // promises, and without it a Speak tapped just after launch showed
            // a false "No offline voice available" even though a voice existed.
            if (initialised.get()) pending = text
            return false
        }
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