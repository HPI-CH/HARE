package sensors_in_paradise.sonar.screen_recording

import org.json.JSONArray
import sensors_in_paradise.sonar.JSONStorage
import java.io.File

class StringItemStorage(file: File) : JSONStorage(file) {
    lateinit var items: JSONArray
    val nonDeletableItems = ArrayList<String>()
    override fun onFileNewlyCreated() {
        json.put("items", JSONArray())
    }

    override fun onJSONInitialized() {
        items = json.getJSONArray("items")
    }

    fun addItemIfNotAdded(item: String, deletable: Boolean = true): Boolean {
        var alreadyAdded = false
        for (i in 0 until items.length()) {
            if (items[i] == item) {
                alreadyAdded = true
                break
            }
        }
        if (!alreadyAdded) {
            addItem(item)
            save()
        }
        if (!deletable) {
            nonDeletableItems.add(item)
        }
        return !alreadyAdded
    }

    fun addItem(item: String) {
        items.put(item)
        save()
    }

    fun removeItem(item: String) {
        for (i in 0 until items.length()) {
            if (items[i] == item) {
                items.remove(i)
                break
            }
        }
       save()
    }

    fun getItemsAsArray(): Array<String> {
        return Array(this.items.length()) { i -> this.items[i].toString() }
    }

	fun getItemsAsArrayList(): ArrayList<String> {
        return getItemsAsArray().toCollection(ArrayList())
    }
}
