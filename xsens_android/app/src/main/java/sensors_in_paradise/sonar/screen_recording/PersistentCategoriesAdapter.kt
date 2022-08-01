package sensors_in_paradise.sonar.screen_recording

import android.annotation.SuppressLint
import android.app.AlertDialog
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.ImageView
import android.widget.RelativeLayout
import android.widget.TextView
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import sensors_in_paradise.sonar.R

class PersistentCategoriesAdapter(private val itemsStorage: CategoryItemStorage) :
    RecyclerView.Adapter<PersistentCategoriesAdapter.ViewHolder>() {
    private var dataSet: List<CategoryItem> = itemsStorage.getItems()
    // private var nestedItems: List<Pair<>> = ArrayList()
    private var filterText: String = ""

    // This is handed over to the nested recyclerviews
    private var onItemClicked: ((value: String) -> Unit)? = null

    inner class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val expandableLayout: RelativeLayout = view.findViewById(R.id.expandable_layout)
        val mTextView: TextView = view.findViewById(R.id.itemTv)
        val mArrowImage: ImageView = view.findViewById(R.id.arrow_imageview)
        val nestedRecyclerView: RecyclerView = view.findViewById(R.id.child_rv)
        val deleteButton: ImageView = view.findViewById(R.id.btn_delete_stringsDialogItem)
        val categoryWrapper: ConstraintLayout = view.findViewById(R.id.category_wrapper)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): PersistentCategoriesAdapter.ViewHolder {
        val view = when (viewType) {
            R.layout.category_item -> {
                LayoutInflater.from(parent.context).inflate(R.layout.category_item, parent, false)
            }
            R.layout.search_results_item -> {
                LayoutInflater.from(parent.context).inflate(R.layout.search_results_item, parent, false)
            }
            else -> {
                LayoutInflater.from(parent.context).inflate(R.layout.add_category_item, parent, false)
            }
        }

        return ViewHolder(view)
    }

    override fun onBindViewHolder(viewHolder: PersistentCategoriesAdapter.ViewHolder, position: Int) {
        if (position == dataSet.size) {
            viewHolder.categoryWrapper.setOnClickListener { showAddCategoryDialog(viewHolder) }
            return
        }

        val model: CategoryItem = getFilteredModelAt(position)

        viewHolder.deleteButton.setOnClickListener {
            itemsStorage.removeCategory(model.itemText)
            if (position != -1) {
                dataSet = itemsStorage.getItems()
                notifyItemRemoved(position)
            }
        }

        val isDeletable = itemsStorage.isCategoryDeletable(model.itemText)
        viewHolder.deleteButton.visibility = if (isDeletable) View.VISIBLE else View.INVISIBLE

        viewHolder.mTextView.text = model.itemText

        val isExpanded: Boolean = model.isExpanded
        viewHolder.expandableLayout.visibility = if (isExpanded) View.VISIBLE else View.GONE

        if (isExpanded) {
            viewHolder.mArrowImage.setImageResource(R.drawable.ic_baseline_keyboard_arrow_up_24)
        } else {
            viewHolder.mArrowImage.setImageResource(R.drawable.ic_baseline_keyboard_arrow_down_24)
        }

        val nestedItems = model.nestedList
        val nestedAdapter = NestedAdapter(nestedItems)
        viewHolder.nestedRecyclerView.layoutManager = GridLayoutManager(viewHolder.itemView.context, 2)
        viewHolder.nestedRecyclerView.setHasFixedSize(true)
        viewHolder.nestedRecyclerView.adapter = nestedAdapter

        nestedAdapter.setOnItemClickedListener { value -> this.onItemClicked?.let { it(value) } }

        nestedAdapter.setOnItemLongClickedListener { value: String, index: Int ->
            itemsStorage.removeEntry(value, dataSet[position].itemText)
            if (index != -1) {
                update(position)
            }
        }

        viewHolder.categoryWrapper.setOnClickListener {
            model.isExpanded = !model.isExpanded
            notifyItemChanged(viewHolder.adapterPosition)
        }
    }

    override fun getItemCount() = getFilteredItemCount()

    override fun getItemViewType(position: Int): Int {
        return when {
            position == dataSet.size -> {
                R.layout.add_category_item // add-new-category button at bottom
            }
            filterText != "" -> {
                R.layout.search_results_item // search-results category
            }
            else -> {
                R.layout.category_item // default category item
            }
        }
    }

    fun setOnItemClickedListener(onItemClicked: (value: String) -> Unit) {
        this.onItemClicked = onItemClicked
    }

    fun updateByCategory(category: String) {
        val index = findIndexByCategory(category)
        if (index != -1) update(index)
    }

    private fun findIndexByCategory(category: String): Int {
        for (i in dataSet.indices) {
            if (dataSet[i].itemText == category) {
                return i
            }
        }
        return -1
    }

    private fun update(position: Int) {
        dataSet = itemsStorage.getItems()
        dataSet[position].isExpanded = true
        notifyItemChanged(position)
    }

    private fun showAddCategoryDialog(viewHolder: ViewHolder) {
        val builder = AlertDialog.Builder(viewHolder.categoryWrapper.context)
        val editText = EditText(viewHolder.categoryWrapper.context)

        builder.setView(editText)
        builder.setMessage("Add new category")
            .setCancelable(true)
            .setPositiveButton("Submit") { _, _ ->
                val newCategory = editText.text.toString()
                itemsStorage.addCategoryIfNotAdded(newCategory)
                dataSet = itemsStorage.getItems()
                notifyItemInserted(dataSet.size - 1)
            }
            .setNegativeButton("Cancel") { dialog, _ ->
                dialog.dismiss()
            }
        val alert = builder.create()
        alert.show()
    }

    /**
     * Search functionality
     */
    @SuppressLint("NotifyDataSetChanged")
    fun filter(text: String) {
        filterText = text
        notifyDataSetChanged()
    }

    private fun getFilteredItemCount(): Int {
        return if (filterText != "") {
            1 // Only need to show one category - search results
        } else {
            dataSet.size + 1
        }
    }

    private fun isItemMatchedByFilter(item: String): Boolean {
        return item.contains(filterText)
    }

    private fun getFilteredModelAt(index: Int): CategoryItem {
        if (filterText == "") return dataSet[index]

        val itemText = "Search results"
        val nestedList = mutableListOf<Pair<String, Boolean>>()

        val categories = itemsStorage.getCategories()
        for (category in categories) {
            val entries = itemsStorage.getEntries(category)
            for ((label, deletable) in entries) {
                if (isItemMatchedByFilter(label)) {
                    nestedList.add(Pair(label, deletable))
                }
            }
        }

        return CategoryItem(itemText, nestedList, true)
    }
}
