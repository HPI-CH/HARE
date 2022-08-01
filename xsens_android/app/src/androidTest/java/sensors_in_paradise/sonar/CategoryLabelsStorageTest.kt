package sensors_in_paradise.sonar

import android.content.Context
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import junit.framework.TestCase.*
import org.junit.Before
import org.junit.Test
import sensors_in_paradise.sonar.screen_recording.CategoryItemStorage
import java.io.File

class CategoryLabelsStorageTest {
    private val appContext: Context = InstrumentationRegistry.getInstrumentation().targetContext
    private val baseDir = File(appContext.cacheDir, "RecordingsTest")

    @Before
    fun init() {
        baseDir.mkdir()
    }

    @Test
    fun testLoadDefaultData() {
        val file = baseDir.resolve("categoryLabels1.json")
        file.delete()
        val storage = CategoryItemStorage(file)
        Log.d(
            "CategoryLabelsStorageTest-testLoadDefaultData",
            "Storage after creation: ${storage.getJsonString(1)}"
        )

        for ((activity, category) in CategoryItemStorage.DEFAULT_ACTIVITIES) {
            Log.d(
                "CategoryLabelsStorageTest-testLoadDefaultData",
                "Asserting that activity $activity is added in category $category"
            )
            assertFalse(storage.isCategoryDeletable(category))
            assertTrue(storage.isEntryAdded(activity, category))
            assertTrue(storage.isEntryAdded(activity))
        }
        file.delete()
    }

    @Test
    fun testSave() {
        val file = baseDir.resolve("categoryLabels2.json")
        file.delete()
        val storage = CategoryItemStorage(file)
        Log.d("CategoryLabelsStorageTest", "Storage after creation: ${storage.getJsonString(1)}")
        storage.addCategoryIfNotAdded("reinigung")
        Log.d(
            "CategoryLabelsStorageTest",
            "Storage after adding category: ${storage.getJsonString(1)}"
        )
        storage.addEntry("wischen", "reinigung", save = true)
        Log.d(
            "CategoryLabelsStorageTest",
            "Storage after adding entry: ${storage.getJsonString(1)}"
        )
        assertTrue(storage.isEntryAdded("wischen", "reinigung"))
        val storage2 = CategoryItemStorage(file)
        Log.d(
            "CategoryLabelsStorageTest",
            "Storage2 after loading from file: ${storage2.getJsonString(1)}"
        )
        assertTrue(storage2.isEntryAdded("wischen", "reinigung"))

        file.delete()
    }
}
