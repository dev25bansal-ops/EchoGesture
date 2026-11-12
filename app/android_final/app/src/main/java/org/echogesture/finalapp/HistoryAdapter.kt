package org.echogesture.finalapp

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.text.DateFormat
import java.util.Date
import java.util.Locale

/** Recent real recognitions, most recent first. No frames are ever stored. */
class HistoryAdapter(
    private val onCopy: (String) -> Unit
) : RecyclerView.Adapter<HistoryAdapter.Row>() {

    private var items: List<RecognitionHistory.Entry> = emptyList()
    private val timeFormat: DateFormat =
        DateFormat.getTimeInstance(DateFormat.SHORT, Locale.getDefault())

    class Row(view: View) : RecyclerView.ViewHolder(view) {
        val label: TextView = view.findViewById(R.id.historyLabel)
        val score: TextView = view.findViewById(R.id.historyScore)
        val time: TextView = view.findViewById(R.id.historyTime)
        val copy: ImageButton = view.findViewById(R.id.historyCopy)
    }

    fun submit(entries: List<RecognitionHistory.Entry>) {
        items = entries
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Row {
        val v = LayoutInflater.from(parent.context)
            .inflate(R.layout.row_history, parent, false)
        return Row(v)
    }

    override fun onBindViewHolder(holder: Row, position: Int) {
        val e = items[position]
        holder.label.text = e.label
        holder.score.text = holder.itemView.context.getString(
            R.string.history_item_score,
            "%.0f".format(e.score * 100f)
        )
        holder.time.text = timeFormat.format(Date(e.timestampMillis))
        holder.copy.setOnClickListener { onCopy(e.label) }
        holder.itemView.contentDescription =
            "${e.label}, ${"%.0f".format(e.score * 100f)} percent"
    }

    override fun getItemCount(): Int = items.size
}