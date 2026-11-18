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
        // Cap the list. With saveHistory on, every add re-serialises the whole
        // list into a single SharedPreferences string on the UI thread, so an
        // unbounded history turns each recognition into O(n) work that grows
        // without limit during a demo or a long session.
        while (entries.size > MAX_ENTRIES) entries.removeAt(entries.size - 1)
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
        // Build into a temp list and skip individual bad rows. The previous
        // version cleared `entries` and then added in a loop, so a single
        // malformed row (a missing "ts"/"score"/"label", e.g. written by an
        // older schema - the stored object carries no version key) threw
        // mid-loop, runCatching swallowed it, and every valid entry AFTER that
        // row was silently dropped. The next add() then persisted the truncated
        // list, making the loss permanent with nothing on screen to signal it.
        val loaded = ArrayList<Entry>()
        runCatching {
            val array = JSONArray(raw)
            for (i in 0 until array.length()) {
                val o = runCatching { array.getJSONObject(i) }.getOrNull() ?: continue
                val label = runCatching { o.getString("label") }.getOrNull() ?: continue
                val score = runCatching { o.getDouble("score").toFloat() }.getOrNull() ?: continue
                val ts = runCatching { o.getLong("ts") }.getOrNull() ?: continue
                loaded.add(Entry(label = label, score = score, timestampMillis = ts))
            }
        }
        entries.clear()
        entries.addAll(loaded)
    }

    private companion object {
        const val KEY_ENTRIES = "entries"

        /** Newest-first cap. Keeps one recognition's persistence cost bounded. */
        const val MAX_ENTRIES = 500
    }
}