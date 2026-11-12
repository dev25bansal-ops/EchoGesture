package org.echogesture.finalapp

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.dialog.MaterialAlertDialogBuilder

/**
 * Recent real recognitions. Session-only unless the user explicitly enabled
 * saving on this device. Copying a label and clearing the list are the only
 * write operations; nothing here can send data anywhere.
 */
class HistoryScreen(
    private val activity: MainActivity,
    private val history: RecognitionHistory,
    private val prefs: AppPreferences
) {

    val root: View = LayoutInflater.from(activity).inflate(R.layout.screen_history, null, false)

    private val list: RecyclerView = root.findViewById(R.id.historyList)
    private val emptyView: View = root.findViewById(R.id.historyEmpty)
    private val subtitle: TextView = root.findViewById(R.id.historySubtitle)
    private val clearButton: Button = root.findViewById(R.id.historyClear)

    private val adapter = HistoryAdapter { label -> copy(label) }
    private var bound = false

    fun onBind() {
        if (!bound) {
            bound = true
            list.layoutManager = LinearLayoutManager(activity)
            list.adapter = adapter
            clearButton.setOnClickListener { confirmClear() }
        }
        refresh()
    }

    fun refresh() {
        val entries = history.snapshot()
        adapter.submit(entries)
        val empty = entries.isEmpty()
        emptyView.visibility = if (empty) View.VISIBLE else View.GONE
        list.visibility = if (empty) View.GONE else View.VISIBLE
        clearButton.visibility = if (empty) View.GONE else View.VISIBLE
        subtitle.text = activity.getString(
            if (prefs.saveHistory) R.string.history_subtitle_persistent
            else R.string.history_subtitle_session
        )
    }

    private fun copy(label: String) {
        val cm = activity.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText(activity.getString(R.string.app_name), label))
        Toast.makeText(activity, R.string.action_copied, Toast.LENGTH_SHORT).show()
    }

    private fun confirmClear() {
        val count = history.size()
        if (count == 0) return
        MaterialAlertDialogBuilder(activity)
            .setTitle(R.string.history_clear)
            .setMessage(activity.getString(R.string.history_clear_confirm, count))
            .setNegativeButton(R.string.action_cancel, null)
            .setPositiveButton(R.string.history_clear) { _, _ ->
                history.clear()
                refresh()
                Toast.makeText(activity, R.string.history_cleared, Toast.LENGTH_SHORT).show()
            }
            .show()
    }
}