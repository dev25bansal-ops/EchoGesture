package org.echogesture.finalapp

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Recognition history produced by real inference. Session-only by default: the
 * list lives in memory and is dropped when the process ends. If the user turns
 * on "keep history on this device" the entries are written to the app's private
 * preferences file — still local, never synced, and never containing frames,
 * images or video. Only the recognised gloss, its model score and a timestamp
 * are stored.
 */
class RecognitionHistory(context: Context) {

    data class Entry(
        val label: String,
        val score: Float,
        val timestampMillis: Long
    )

    private val appContext = context.applicationContext
    private val prefs = appContext.getSharedPreferences("echogesture_history", Context.MODE_PRIVATE)
    private val entries = CopyOnWriteArrayList<Entry>()
    private val listeners = CopyOnWriteArrayList<() -> Unit>()

    val isPersistent: Boolean
        get() = appContext.getSharedPreferences(AppPreferences.PREFS_FILE, Context.MODE_PRIVATE)
            .getBoolean("save_history", false)

    init {
        if (isPersistent) load()
    }

    fun snapshot(): List<Entry> = entries.toList()

    fun size(): Int = entries.size

    fun mostRecentLabels(): Set<String> = entries.map { it.label }.toSet()

    fun add(label: String, score: Float, timestampMillis: Long = System.currentTimeMillis()) {
        entries.add(0, Entry(label, score, timestampMillis))
        persistIfEnabled()
        notifyChanged()
    }

    fun clear() {
        entries.clear()
        prefs.edit().remove(KEY_ENTRIES).apply()
        notifyChanged()
    }

    /** Re-reads persisted entries. Call when the persistence setting changes. */
    fun onPersistenceChanged(enabled: Boolean) {
        if (enabled) {
            persistIfEnabled()
        } else {
            prefs.edit().remove(KEY_ENTRIES).apply()
        }
    }

    fun addListener(listener: () -> Unit) {
        listeners.add(listener)
    }

    private fun notifyChanged() = listeners.forEach { runCatching(it) }

    private fun persistIfEnabled() {
        if (!isPersistent) return
        val array = JSONArray()
        entries.forEach { e ->
            array.put(
                JSONObject()
                    .put("label", e.label)
                    .put("score", e.score.toDouble())
                    .put("ts", e.timestampMillis)
            )
        }
        prefs.edit().putString(KEY_ENTRIES, array.toString()).apply()
    }

    private fun load() {
        val raw = prefs.getString(KEY_ENTRIES, null) ?: return
        runCatching {
            val array = JSONArray(raw)
            entries.clear()
            for (i in 0 until array.length()) {
                val o = array.getJSONObject(i)
                entries.add(
                    Entry(
                        label = o.getString("label"),
                        score = o.getDouble("score").toFloat(),
                        timestampMillis = o.getLong("ts")
                    )
                )
            }
        }
    }

    private companion object {
        const val KEY_ENTRIES = "entries"
    }
}