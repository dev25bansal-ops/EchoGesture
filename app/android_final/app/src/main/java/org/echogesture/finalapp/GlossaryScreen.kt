package org.echogesture.finalapp

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomsheet.BottomSheetDialog

/**
 * Browsable representation of the model's real 100-label vocabulary. The list
 * is the actual class list the classifier emits; nothing is invented and no
 * teaching content is implied.
 */
class GlossaryScreen(
    private val activity: MainActivity,
    private val history: RecognitionHistory,
    private val speaker: Speaker
) {

    val root: View = LayoutInflater.from(activity).inflate(R.layout.screen_glossary, null, false)

    private val list: RecyclerView = root.findViewById(R.id.glossaryList)
    private val search: EditText = root.findViewById(R.id.glossarySearch)
    private val clearSearch: ImageButton = root.findViewById(R.id.glossaryClearSearch)
    private val countChip: TextView = root.findViewById(R.id.glossaryCount)
    private val subtitle: TextView = root.findViewById(R.id.glossarySubtitle)
    private val emptyView: View = root.findViewById(R.id.glossaryEmpty)
    private val emptyText: TextView = root.findViewById(R.id.glossaryEmptyText)

    private val adapter = GlossaryAdapter { label -> openDetail(label) }
    private var labels: List<String> = emptyList()
    private var bound = false

    fun onBind() {
        if (!bound) {
            bound = true
            list.layoutManager = LinearLayoutManager(activity)
            list.adapter = adapter
            search.addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun afterTextChanged(s: Editable?) {
                    applyQuery(s?.toString().orEmpty())
                }
            })
            clearSearch.setOnClickListener {
                search.setText("")
                search.clearFocus()
            }
        }
        // Re-read the seen set each time the tab is opened so the "recognised
        // this session" marker reflects the current history.
        adapter.setRecentlySeen(history.mostRecentLabels())
    }

    fun submitLabels(all: List<String>) {
        labels = all
        adapter.submit(all)
        subtitle.text = activity.getString(R.string.glossary_subtitle, all.size)
        updateCount()
    }

    fun onHistoryChanged() {
        adapter.setRecentlySeen(history.mostRecentLabels())
    }

    private fun applyQuery(text: String) {
        adapter.filter(text)
        clearSearch.visibility = if (text.isBlank()) View.GONE else View.VISIBLE
        updateCount()
        emptyText.text = activity.getString(R.string.glossary_empty, text)
        emptyView.visibility = if (adapter.shownCount == 0) View.VISIBLE else View.GONE
        list.visibility = if (adapter.shownCount == 0) View.GONE else View.VISIBLE
    }

    private fun updateCount() {
        countChip.text = activity.getString(
            R.string.glossary_count,
            adapter.shownCount,
            labels.size
        )
    }

    private fun openDetail(label: String) {
        val dialog = BottomSheetDialog(activity)
        val view = LayoutInflater.from(activity).inflate(R.layout.sheet_gloss, null, false)
        view.findViewById<TextView>(R.id.sheetGlossLabel).text = label
        view.findViewById<TextView>(R.id.sheetGlossMeta).text =
            activity.getString(R.string.glossary_subtitle, labels.size)
        view.findViewById<Button>(R.id.sheetGlossCopy).setOnClickListener {
            val cm = activity.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText(activity.getString(R.string.app_name), label))
            Toast.makeText(activity, R.string.action_copied, Toast.LENGTH_SHORT).show()
        }
        view.findViewById<Button>(R.id.sheetGlossSpeak).setOnClickListener {
            if (!speaker.speak(label)) {
                Toast.makeText(activity, R.string.tts_unavailable, Toast.LENGTH_SHORT).show()
            }
        }
        dialog.setContentView(view)
        dialog.show()
    }
}