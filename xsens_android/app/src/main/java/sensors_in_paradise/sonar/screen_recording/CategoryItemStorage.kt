package sensors_in_paradise.sonar.screen_recording

import org.json.JSONObject
import sensors_in_paradise.sonar.GlobalValues
import sensors_in_paradise.sonar.JSONStorage
import java.io.File

/**
Example Structure:
    { "labels": {
        "Mittag": {
                "deletable": false
                "entries": {
                "Essen bringen": {"deletable": false},
                "Füttern": {"deletable": false}
            }
        },
        "Körperpflege": {
                "deletable": true
                "entries": {
                "Gesamtwäsche am Bett": {"deletable": false },
                "Haare kämmen": {"deletable": false }
            }
        },
    },
    "version": 2
    }

Legacy:
    {"items": [
        {
            "category": "Mittag",
            "entries": ["Essen bringen", "Füttern"]
        },
        {
            "category": "Waschen",
            "entries": ["OK waschen", "Haare kämmen"]
        },
]}
 */

class CategoryItemStorage(file: File) : JSONStorage(file) {
    private lateinit var labelsData: JSONObject

    override fun onFileNewlyCreated() {
        labelsData = JSONObject()
        json.put("labels", labelsData)
        json.put("version", 2)
        addDefaultActivities()
    }

    override fun onJSONInitialized() {
        labelsData = json.getJSONObject("labels")
        if (isLegacyJson()) {
            convertFromLegacyJson()
        }
    }

    private fun isLegacyJson(): Boolean {
        return json.optInt("version") < 2
    }

    private fun convertFromLegacyJson() {
        val items = json.getJSONArray("items")
        for (i in 0 until items.length()) {
            val item = items.getJSONObject(i)
            val category = item.getString("category")
            val entries = item.getJSONArray("entries")
            addCategoryIfNotAdded(category, deletable = false, save = false)
            for (j in 0 until entries.length()) {
                addEntry(entries.getString(j), category, deletable = false, save = false)
            }
        }
        json.remove("items")
        save()
    }

    private fun addDefaultActivities() {
        for ((activity, category) in DEFAULT_ACTIVITIES) {
            addEntryIfNotAdded(activity, category, false)
        }
    }

    fun addEntryIfNotAdded(
        entry: String,
        category: String = DEFAULT_CATEGORY,
        deletable: Boolean = true,
        save: Boolean = true
    ): Boolean {
        val alreadyAdded = isEntryAdded(entry, category)

        if (!alreadyAdded) {
            addCategoryIfNotAdded(category, deletable, save)
            addEntry(entry, category, deletable, save)
        }
        return !alreadyAdded
    }

    fun addCategoryIfNotAdded(category: String, deletable: Boolean = true, save: Boolean = true) {
        if (!hasCategory(category)) {
            addCategory(category, deletable, save)
        }
    }

    /**Returns the category json object*/
    private fun addCategory(
        category: String,
        deletable: Boolean = true,
        save: Boolean
    ): JSONObject {
        val obj = JSONObject().apply {
            put("entries", JSONObject())
            put("deletable", deletable)
        }
        labelsData.put(category, obj)
        if (save) {
            save()
        }
        return obj
    }

    private fun hasCategory(category: String): Boolean {
        return labelsData.has(category)
    }

    // Removes all entries in the category as well
    fun removeCategory(category: String) {
        labelsData.remove(category)
        save()
    }

    fun getItems(): MutableList<CategoryItem> {
        val itemList = mutableListOf<CategoryItem>()

        val categories = getCategories()
        for (category in categories) {
            val entries = getEntries(category)
            val item = CategoryItem(category, entries)
            itemList.add(item)
        }

        return itemList
    }

    fun getCategories(): ArrayList<String> {
        val result = ArrayList<String>()
        for (category in labelsData.keys()) {
            result.add(category)
        }
        return result
    }

    fun addEntry(
        entry: String,
        category: String = DEFAULT_CATEGORY,
        deletable: Boolean = true,
        save: Boolean
    ) {
        val jsonObj = findJSONObjectOfCategory(category) ?: addCategory(category, deletable, false)
        val entries = jsonObj.getJSONObject("entries")
        entries.put(entry, JSONObject().apply { put("deletable", deletable) })
        if (save) {
            save()
        }
    }

    fun removeEntry(entry: String, category: String = DEFAULT_CATEGORY) {
        val jsonObj = findJSONObjectOfCategory(category)
        if (jsonObj != null) {
            val entries = jsonObj.getJSONObject("entries")
            entries.remove(entry)
            save()
        }
    }

    /* Returns the labels in the given category as well as if they are deletable or not
    * */
    fun getEntries(category: String): ArrayList<Pair<String, Boolean>> {
        val jsonObj = findJSONObjectOfCategory(category)
        val result = ArrayList<Pair<String, Boolean>>()
        if (jsonObj != null) {
            val entries = jsonObj.getJSONObject("entries")
            val labels = entries.keys()
            for (label in labels) {
                val deletable = entries.getJSONObject(label).getBoolean("deletable")
                result.add(Pair(label, deletable))
            }
        }
        return result
    }

    fun isEntryAdded(entry: String, category: String? = null): Boolean {
        if (category == null) {
            for (cat in labelsData.keys()) {
                val categoryObj = labelsData.optJSONObject(cat)
                val entries = categoryObj?.getJSONObject("entries")
                if (entries?.has(entry) == true) {
                    return true
                }
            }
        } else {
            val categoryObj = labelsData.optJSONObject(category)
            val entries = categoryObj?.getJSONObject("entries")
            return entries?.has(entry) ?: false
        }
        return false
    }

    private fun findJSONObjectOfCategory(category: String): JSONObject? {
        return labelsData.optJSONObject(category)
    }

    fun isCategoryDeletable(category: String): Boolean {
        return findJSONObjectOfCategory(category)?.getBoolean("deletable") ?: true
    }

    companion object {
        val DEFAULT_ACTIVITIES = linkedMapOf(
            GlobalValues.NULL_ACTIVITY to GlobalValues.OTHERS_CATEGORY,
        )
        const val DEFAULT_CATEGORY = GlobalValues.OTHERS_CATEGORY
    }
}
