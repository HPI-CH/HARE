package sensors_in_paradise.sonar.use_cases

import android.content.Context
import android.widget.Toast
import sensors_in_paradise.sonar.R
import sensors_in_paradise.sonar.util.dialogs.MessageDialog
import java.io.File
import java.io.IOException
import java.nio.file.*
import java.nio.file.attribute.BasicFileAttributes

class UseCase(
    val context: Context,
    private val baseDir: File,
    val title: String,
    private val onUseCaseDeleted: (useCase: UseCase) -> Unit,
    private val onUseCaseDuplicated: (originalUseCase: UseCase, duplicateTitle: String) -> UseCase
) {
    private val useCaseDir = baseDir.resolve(title).apply { mkdirs() }
    private val useCaseStorage = UseCaseStorage(useCaseDir.resolve(STORAGE_SUB_DIR_NAME))
    private var recordingsSubDir =
        getRecordingsDir().resolve(useCaseStorage.getSelectedSubDir()).apply { mkdir() }

    fun getActivityLabelsJSONFile(): File {
        return useCaseDir.resolve("labels.json")
    }

    fun getPeopleJSONFile(): File {
        return useCaseDir.resolve("people.json")
    }

    fun getModelFile(): File {
        return useCaseDir.resolve(MODEL_FILE_NAME)
    }

    private fun getRecordingsDir(): File {
        return useCaseDir.resolve("recordings").apply { mkdir() }
    }

    fun getTrainingHistoryJSONFile(): File {
        return useCaseDir.resolve("trainHistory.json")
    }

    private fun getPredictionsDir(): File {
        return useCaseDir.resolve("predictions").apply { mkdir() }
    }

    fun getModelCheckpointsDir(): File {
        return useCaseDir.resolve("modelCheckpoints").apply { mkdir() }
    }

    fun getPredictionHistoryJSONFile(startTimestamp: Long): File {
        val predictionsDir = getPredictionsDir()
        return predictionsDir.resolve("$startTimestamp.json")
    }

    fun getAvailableRecordingSubDirs(): List<String> {
        val subDirs =
            (getRecordingsDir().listFiles { d, name -> File(d, name).isDirectory })?.map { it.name }
        if (subDirs == null) {
            return arrayListOf(recordingsSubDir.apply { mkdir() }.name)
        }
        return subDirs
    }

    fun setRecordingsSubDir(dir: String) {
        recordingsSubDir = getRecordingsDir().resolve(dir).apply { mkdir() }
        useCaseStorage.setSelectedSubDir(dir)
    }

    fun getRecordingsSubDir(): File {
        return recordingsSubDir
    }

    fun getRelativePathOfRecordingsSubDir(): String {
        return recordingsSubDir.absolutePath.replaceFirst(baseDir.absolutePath, "")
    }

    fun importDefaultModel() {
        extractFileFromAssets(context, "resnet_model.tflite", getModelFile())
    }

    fun getDisplayInfo(): String {
        return "\uD83D\uDCBC " + title + "    \uD83D\uDCC2 " + recordingsSubDir.name
    }

    fun delete() {
        try {
            delete(useCaseDir)
        } catch (e: IOException) {
            Toast.makeText(context, "Deleting file failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
        onUseCaseDeleted(this)
    }

    fun duplicate(titleOfDuplicate: String): UseCase {
        val targetDir = useCaseDir.parentFile!!.resolve(titleOfDuplicate)
        copyFolder(useCaseDir.toPath(), targetDir.toPath(), StandardCopyOption.REPLACE_EXISTING)
        return onUseCaseDuplicated(this, titleOfDuplicate)
    }

    @Throws(IOException::class)
    private fun delete(f: File) {
        if (f.isDirectory) {
            val list = f.listFiles()
            if (list != null) {
                for (c in list) delete(c)
            }
        }
        if (!f.delete()) throw IOException("Failed to delete file: $f")
    }

    fun showNoModelFileExistingDialog(context: Context) {
        MessageDialog(
            context,
            message = context.getString(
                R.string.missing_model_dialog_message,
                getModelFile().absolutePath
            ),
            title = context.getString(R.string.missing_model_dialog_title),
            positiveButtonText = "Okay",
            neutralButtonText = "Import default Model",
            onNeutralButtonClickListener = { _, _ ->
                importDefaultModel()
            }
        )
    }

    companion object {
        const val DEFAULT_RECORDINGS_SUB_DIR_NAME = "default"
        const val STORAGE_SUB_DIR_NAME = "storage.json"
        const val MODEL_FILE_NAME = "model.tflite"
        fun extractFileFromAssets(context: Context, assetFileName: String, targetFile: File) {
            val iStream = context.assets.open(assetFileName)
            iStream.use { input ->
                targetFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        @Suppress("SpreadOperator")
        @Throws(IOException::class)
        private fun copyFolder(source: Path, target: Path, vararg options: CopyOption?) {
            Files.walkFileTree(source, object : SimpleFileVisitor<Path?>() {
                @Throws(IOException::class)
                override fun preVisitDirectory(dir: Path?, attrs: BasicFileAttributes?): FileVisitResult {
                    Files.createDirectories(target.resolve(source.relativize(dir)))
                    return FileVisitResult.CONTINUE
                }

                @Throws(IOException::class)
                override fun visitFile(file: Path?, attrs: BasicFileAttributes?): FileVisitResult {
                    if (file != null) {
                        Files.copy(file, target.resolve(source.relativize(file)), *options)
                    }
                    return FileVisitResult.CONTINUE
                }
            })
        }
    }
}
