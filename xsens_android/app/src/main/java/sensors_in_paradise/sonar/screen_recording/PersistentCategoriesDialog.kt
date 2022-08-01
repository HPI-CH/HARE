package sensors_in_paradise.sonar.screen_recording

import android.app.AlertDialog
import android.content.Context
import android.view.LayoutInflater
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import androidx.core.widget.addTextChangedListener
import androidx.recyclerview.widget.RecyclerView
import sensors_in_paradise.sonar.R
import java.io.File
import java.util.*

class PersistentCategoriesDialog(
    val context: Context,
    title: String,
    storageFile: File,
    callback: (value: String) -> Unit = {},
) {
    var dialog: AlertDialog
    private val storage = CategoryItemStorage(storageFile)
    var adapter: PersistentCategoriesAdapter

    init {

        val builder: AlertDialog.Builder = AlertDialog.Builder(context)
        builder.setTitle(title)

        val root = LayoutInflater.from(context).inflate(R.layout.string_array_dialog, null)
        val searchEditText = root.findViewById<EditText>(R.id.editText_search_stringArrayDialog)
        val rv = root.findViewById<RecyclerView>(R.id.recyclerView_items_stringArrayDialog)

        adapter = PersistentCategoriesAdapter(storage)
        rv.adapter = adapter
        builder.setView(root)
        builder.setPositiveButton("Add new item", null)

        dialog = builder.create()
        adapter.setOnItemClickedListener { value ->
            callback(value)
            dialog.dismiss()
        }

        dialog.setOnShowListener {
            val button: Button =
                dialog.getButton(AlertDialog.BUTTON_POSITIVE)
            button.isEnabled = false
            button.setOnClickListener { showCategorySelectionDialog(searchEditText) }
            searchEditText.addTextChangedListener { text ->
                button.isEnabled = (!isItemAlreadyAdded(text.toString()) && text.toString() != "")
                adapter.filter(text.toString())
            }
        }
    }

    fun show(cancelable: Boolean = true) {
        dialog.setCancelable(cancelable)
        dialog.show()
    }

    private fun isItemAlreadyAdded(entry: String): Boolean {
        return storage.isEntryAdded(entry)
    }

    private fun showCategorySelectionDialog(searchEditText: EditText) {
        val builder = AlertDialog.Builder(context)
        val spinner = Spinner(context)
        val spinnerAdapter = ArrayAdapter(context, android.R.layout.simple_dropdown_item_1line,
            storage.getCategories())
        spinner.adapter = spinnerAdapter

        builder.setView(spinner)
        builder.setMessage("Choose category")
            .setCancelable(true)
            .setPositiveButton("Submit") { _, _ ->
                val category = spinner.selectedItem.toString()
                storage.addEntryIfNotAdded(searchEditText.text.toString().lowercase(Locale.getDefault()), category)
                adapter.updateByCategory(category)
                searchEditText.setText("")
            }
            .setNegativeButton("Cancel") { dialog, _ ->
                dialog.dismiss()
            }
        val alert = builder.create()
        alert.show()
    }
}
