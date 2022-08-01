package sensors_in_paradise.sonar
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import sensors_in_paradise.sonar.uploader.FileUploader
import java.io.File

class FileUploaderTest {
    private val rootDir = File(".").resolve("testTempDir")
    private val uploader = FileUploader(rootDir, null)

    @Before
    fun init() {
        rootDir.mkdir()
        val dirA = rootDir.resolve("dirA0").resolve("dirA00").resolve("dirA000")
        val dirB = rootDir.resolve("dirB0").resolve("dirB00").resolve("dirB000")
        dirA.mkdirs()
        dirB.mkdirs()

        val dirA001 = dirA.parentFile.resolve("dirA001")
        dirA001.mkdir()

        dirA001.resolve("dirA001_f0.csv").createNewFile()
        dirA001.resolve("dirA001_f1.csv").createNewFile()
        dirB.resolve("dirB_f0.csv").createNewFile()
        dirA.resolve("dirB_f0.csv").createNewFile()
    }

    @Test
    fun resolveSuffixFromFile() {
        val testFile = rootDir.resolve("dirC0").resolve("dirC00").resolve("dirC000").resolve("dirC_f0.csv")
        assertEquals(uploader.getURLSuffixForFile(testFile), "dirC0/dirC00/dirC000/dirC_f0.csv")
    }

    @Test
    fun detectFilesToBeUploaded() {
        val detectedFiles = uploader.getFilesToBeUploaded(rootDir)
        assertEquals(detectedFiles.size, 4)
    }

    @After
    fun cleanUp() {
        rootDir.deleteRecursively()
    }
}
