package sensors_in_paradise.sonar

import androidx.test.platform.app.InstrumentationRegistry
import junit.framework.TestCase.*
import org.junit.Before
import org.junit.Test
import java.io.File

class JSONStorageTest {
    val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    val file = File(appContext.cacheDir, "test.json")
    @Before
    fun init() {
        file.delete()
    }

    @Test
    fun createNewStorageTest() {
        assertFalse(file.exists())
        object : JSONStorage(file) {
            override fun onFileNewlyCreated() {}
            override fun onJSONInitialized() {}
        }
        assertTrue(file.exists())
        file.delete()
    }

    @Test
    fun changeDataStorageTest() {
        assertFalse(file.exists())
        val storageA = object : JSONStorage(file) {
            override fun onFileNewlyCreated() {}
            override fun onJSONInitialized() {}
        }
        storageA.json.put("testKey", "testValue")
        storageA.save()
        assertTrue(file.exists())

        val storageB = object : JSONStorage(file) {
            override fun onFileNewlyCreated() {}
            override fun onJSONInitialized() {}
        }
        assertTrue(storageB.json.has("testKey"))
        assertEquals(storageB.json.getString("testKey"), "testValue")
        file.delete()
    }
}
