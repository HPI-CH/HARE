package sensors_in_paradise.sonar.screen_recording

import android.app.AlertDialog
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.card.MaterialCardView

class NestedAdapter(private val entries: List<Pair<String, Boolean>>) :
    RecyclerView.Adapter<NestedAdapter.NestedViewHolder>() {
    private var onItemClicked: ((value: String) -> Unit)? = null
    private var onItemLongClicked: ((value: String, position: Int) -> Unit)? = null

    inner class NestedViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val entry: TextView = view.findViewById(sensors_in_paradise.sonar.R.id.nestedItemTv)
        val wrapper: MaterialCardView = view.findViewById(sensors_in_paradise.sonar.R.id.cardView)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): NestedViewHolder {
        val view =
            LayoutInflater.from(parent.context).inflate(
                sensors_in_paradise.sonar.R.layout.nested_item, parent, false)
        return NestedViewHolder(view)
    }

    override fun onBindViewHolder(viewHolder: NestedViewHolder, position: Int) {
        val item = entries[position]
        val entryText = item.first
        val isDeletable = item.second

        viewHolder.entry.text = entryText
        viewHolder.wrapper.setOnClickListener {
            onItemClicked?.let { it1 -> it1(entryText) }
        }

        if (isDeletable) {
            viewHolder.wrapper.strokeColor = ContextCompat.getColor(viewHolder.wrapper.context,
                sensors_in_paradise.sonar.R.color.colorAccent)
            viewHolder.wrapper.setOnLongClickListener {
                val builder = AlertDialog.Builder(viewHolder.wrapper.context)
                builder.setMessage("Are you sure you want to delete?")
                    .setCancelable(false)
                    .setPositiveButton("Yes") { _, _ ->
                        onItemLongClicked?.let { it1 -> it1(entryText, position) }
                    }
                    .setNegativeButton("No") { dialog, _ ->
                        dialog.dismiss()
                    }
                val alert = builder.create()
                alert.show()

                true
            }
        }
    }

    override fun getItemCount(): Int {
        return entries.size
    }

    fun setOnItemClickedListener(onItemClicked: (value: String) -> Unit) {
        this.onItemClicked = onItemClicked
    }

    fun setOnItemLongClickedListener(onItemLongClicked: (value: String, position: Int) -> Unit) {
        this.onItemLongClicked = onItemLongClicked
    }
}
