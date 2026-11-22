package org.echogesture.finalapp

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.util.Log
import androidx.annotation.MainThread
import java.io.IOException

/** Plays the bundled pronunciation of a recognised vocabulary item. */
@MainThread
class Speaker(context: Context) {
    private val assets = context.applicationContext.assets
    private val labels by lazy {
        assets.open(ModelContract.LABELS_ASSET).bufferedReader().use {
            ModelContract.parseLabels(it.readText())
        }
    }
    private var player: MediaPlayer? = null

    fun speak(label: String, onFailure: () -> Unit = {}) {
        shutdown()
        val index = labels.indexOf(label)
        if (index < 0) {
            onFailure()
            return
        }
        val playback = MediaPlayer()
        player = playback
        try {
            playback.setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            assets.openFd("pronunciations/$index.mp3").use { file ->
                playback.setDataSource(file.fileDescriptor, file.startOffset, file.length)
            }
            playback.setOnPreparedListener {
                if (player === it) {
                    try {
                        it.start()
                    } catch (failure: IllegalStateException) {
                        Log.w(TAG, "Could not start pronunciation playback", failure)
                        shutdown()
                        onFailure()
                    }
                }
            }
            playback.setOnCompletionListener {
                if (player === it) shutdown()
            }
            playback.setOnErrorListener { failed, what, extra ->
                if (player === failed) {
                    Log.w(TAG, "Pronunciation playback failed ($what, $extra)")
                    shutdown()
                    onFailure()
                }
                true
            }
            playback.prepareAsync()
        } catch (failure: IOException) {
            Log.w(TAG, "Could not open bundled pronunciation", failure)
            shutdown()
            onFailure()
        } catch (failure: RuntimeException) {
            Log.w(TAG, "Could not prepare playback", failure)
            shutdown()
            onFailure()
        }
    }

    fun shutdown() {
        val active = player
        player = null
        active?.release()
    }

    private companion object {
        const val TAG = "Speaker"
    }
}
