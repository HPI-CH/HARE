package sensors_in_paradise.sonar.history

import android.annotation.SuppressLint
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import sensors_in_paradise.sonar.R

abstract class HistoryAdapter(private val isLineCentered: Boolean = false) :
    RecyclerView.Adapter<HistoryAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val middleLayout: LinearLayout = view.findViewById(R.id.linearLayout_middle_historyItem)
        val middlePrefixTV: TextView = view.findViewById(R.id.tv_middlePrefix_historyItem)
        val titleTV: TextView = view.findViewById(R.id.tv_title_historyItem)
        val subTitleTV: TextView = view.findViewById(R.id.tv_subtitle_historyItem)
        val upperSegmentView: View = view.findViewById(R.id.view_upperSegment_historyItem)
        val lowerSegmentView: View = view.findViewById(R.id.view_lowerSegment_historyItem)
        val middleDotView: View = view.findViewById(R.id.view_middleDot_historyItem)

        val upperDotLayout: LinearLayout = view.findViewById(R.id.linearLayout_upperDot_historyItem)
        val upperPrefixTV: TextView = view.findViewById(R.id.tv_upperPrefix_historyItem)
        val upperSegmentUpperLayoutView: View =
            view.findViewById(R.id.view_upperSegmentUpperLayout_historyItem)
        val lowerSegmentUpperLayoutView: View =
            view.findViewById(R.id.view_lowerSegmentUpperLayout_historyItem)
    }

    override fun onCreateViewHolder(
        parent: ViewGroup,
        viewType: Int
    ): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.history_item, parent, false)
        return ViewHolder(view)
    }

    protected open fun getPrefixOfItem(position: Int): String? {
        return null
    }

    abstract fun getTitleOfItem(position: Int): String

    abstract fun getSubtitleOfItem(position: Int): String

    @SuppressLint("SetTextI18n")
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.apply {
            val prefix = getPrefixOfItem(position)
            if (prefix != null) {
                val prefixTV = if (isLineCentered) upperPrefixTV else middlePrefixTV
                prefixTV.text = prefix
            } else {
                middlePrefixTV.visibility = View.GONE
                upperPrefixTV.visibility = View.GONE
            }
            titleTV.text = getTitleOfItem(position)
            subTitleTV.text = getSubtitleOfItem(position)

            if (!isLineCentered) {
                upperDotLayout.visibility = View.GONE

                val titleParam = titleTV.layoutParams as ViewGroup.MarginLayoutParams
                titleParam.setMargins(0, 8, 0, 0)
                titleTV.layoutParams = titleParam
                val subTitleParam = subTitleTV.layoutParams as ViewGroup.MarginLayoutParams
                subTitleParam.setMargins(0, 0, 0, 8)
                subTitleTV.layoutParams = subTitleParam

                lowerSegmentView.visibility =
                    if (position == itemCount - 1) View.INVISIBLE else View.VISIBLE
                upperSegmentView.visibility = if (position == 0) View.INVISIBLE else View.VISIBLE
            } else {
                middleDotView.visibility = View.INVISIBLE
                middleLayout.visibility =
                    if (position == itemCount - 1) View.GONE else View.VISIBLE

                lowerSegmentUpperLayoutView.visibility =
                    if (position == itemCount - 1) View.INVISIBLE else View.VISIBLE
                upperSegmentUpperLayoutView.visibility =
                    if (position == 0) View.INVISIBLE else View.VISIBLE
            }
        }
    }

    abstract override fun getItemCount(): Int

    /* Use this function instead of notifyItemInserted:
        With this adapter, multiple adjacent items need be re-rendered when a new item is added,
        so that the time beam stays correct
    */
    fun notifyItemAdded(position: Int) {
        notifyItemInserted(position)
        if (itemCount >= 2) {
            when (position) {
                0 -> {
                    notifyItemChanged(position + 1)
                }
                itemCount - 1 -> {
                    notifyItemChanged(position - 1)
                }
            }
        }
    }
}
