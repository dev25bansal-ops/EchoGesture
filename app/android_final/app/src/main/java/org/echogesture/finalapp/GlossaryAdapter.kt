package org.echogesture.finalapp

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

/** Alphabetical browser over the model's real 100-label vocabulary. */
class GlossaryAdapter(
    private val onPick: (String) -> Unit
) : RecyclerView.Adapter<GlossaryAdapter.Row>() {

    private var all: List<String> = emptyList()
    private var shown: List<String> = emptyList()
    private var query: String = ""
    private var recentlySeen: Set<String> = emptySet()

    class Row(view: View) : RecyclerView.ViewHolder(view) {
        val label: TextView = view.findViewById(R.id.glossLabel)
        val seen: ImageView = view.findViewById(R.id.glossSeen)
    }

    fun submit(labels: List<String>) {
        all = labels.sorted()
        applyFilter()
    }

    fun setRecentlySeen(labels: Set<String>) {
        recentlySeen = labels
        notifyDataSetChanged()
    }

    fun filter(text: String) {
        query = text
        applyFilter()
    }

    private fun applyFilter() {
        shown = if (query.isBlank()) {
            all
        } else {
            val needle = normalise(query)
            all.filter { normalise(it).contains(needle) }
        }
        notifyDataSetChanged()
    }

    /**
     * Model labels carry a class suffix and a slash form, e.g. "HURDLE/TRIP1".
     * Search should match what a person types ("hurdle", "trip"), so digits and
     * punctuation are stripped from both sides. The label itself is never
     * rewritten — the real class name is what the model outputs.
     */
    private fun normalise(s: String): String =
        s.lowercase().filter { it.isLetter() }

    val shownCount: Int get() = shown.size

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Row {
        val v = LayoutInflater.from(parent.context)
            .inflate(R.layout.row_gloss, parent, false)
        return Row(v)
    }

    override fun onBindViewHolder(holder: Row, position: Int) {
        val label = shown[position]
        holder.label.text = label
        holder.seen.visibility =
            if (label in recentlySeen) View.VISIBLE else View.GONE
        holder.itemView.contentDescription = label
        holder.itemView.setOnClickListener { onPick(label) }
    }

    override fun getItemCount(): Int = shown.size
}