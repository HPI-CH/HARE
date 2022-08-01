package sensors_in_paradise.sonar.uploader

import sensors_in_paradise.sonar.GlobalValues
import sensors_in_paradise.sonar.screen_recording.Recording
import java.io.File
import java.lang.Exception

class RecordingUIItem(recording: Recording, private val recordingsBaseDir: File) :
    Recording(recording) {
    var error: Exception? = null
    val filesAndDirsToBeUploaded = getFilesAndDirsToBeUploadedList()
    val dirsToBeCreated = filesAndDirsToBeUploaded.filter { file -> file.isDirectory }
    val filesToBeUploaded = filesAndDirsToBeUploaded.filter { file -> file.isFile }
    val label = "${metadataStorage.getPerson()} - ${
        metadataStorage.getActivities().joinToString { it.activity }
    }"
    private val filesAndDirsUploadStatus: MutableMap<File, UploadStatus> =
        getFilesAndDirsToBeUploadedList().associateWith { UploadStatus.NOT_UPLOADED } as MutableMap<File, UploadStatus>

    private fun getFilesAndDirsToBeUploadedList(): ArrayList<File> {
        val files = ArrayList<File>()
        files.addAll(getRecordingFiles())
        files.add(metadataStorage.file)
        if (hasVideoRecording()) {
            files.add(getVideoFile())
        }
        var dirToBeCreated = dir
        while (dirToBeCreated != recordingsBaseDir) {
            files.add(0, dirToBeCreated)
            dirToBeCreated = dirToBeCreated.parentFile ?: recordingsBaseDir
        }
        return files
    }

    fun getSummarizedStatus(): UploadStatus {
        var areAllUploaded = true
        for ((_, status) in filesAndDirsUploadStatus) {
            if (status != UploadStatus.UPLOADED) {
                areAllUploaded = false
                break
            }
        }
        if (areAllUploaded) {
            return UploadStatus.UPLOADED
        }
        for ((_, status) in filesAndDirsUploadStatus) {
            if (status == UploadStatus.UPLOADING) {
                return UploadStatus.UPLOADING
            }
        }

        for ((_, status) in filesAndDirsUploadStatus) {
            if (status == UploadStatus.FAILED) {
                return UploadStatus.FAILED
            }
        }
        var areAllWaiting = true
        for ((_, status) in filesAndDirsUploadStatus) {
            if (status != UploadStatus.NOT_UPLOADED) {
                areAllWaiting = false
                break
            }
        }
        if (areAllWaiting) {
            return UploadStatus.NOT_UPLOADED
        }
        return UploadStatus.UNKNOWN
    }

    fun getStatusLabel(): String {
        var result = ""
        var indent = 0
        for (dir in dirsToBeCreated) {
            result += " ".repeat(indent * 4) + "${getEmojiStatusOfFileOrDir(dir)} \uD83D\uDCC1${dir.name}\n"
            indent++
        }
        for (file in filesToBeUploaded) {
            val docItem = GlobalValues.getFileEmoji(file)
            result += " ".repeat(indent * 4) + "${getEmojiStatusOfFileOrDir(file)}  $docItem${file.name}\n"
        }
        return result
    }

    fun getEmojiStatusOfFileOrDir(file: File): String {
        return statusEmoji(getStatusOfFileOrDir(file))
    }

    fun setStatusOfFileOrDir(fileOrDir: File, status: UploadStatus) {
        filesAndDirsUploadStatus[fileOrDir] = status
    }

    fun getStatusOfFileOrDir(file: File): UploadStatus {
        if (filesAndDirsUploadStatus.containsKey(file)) {
            return filesAndDirsUploadStatus[file]!!
        }
        return UploadStatus.UNKNOWN
    }

    fun getStatusLabelOfFileOrDir(file: File): String {
        return statusLabel(getStatusOfFileOrDir(file))
    }

    fun isUploaded(): Boolean {

        for (file in getRecordingFiles()) {
            if (getStatusOfFileOrDir(file) != UploadStatus.UPLOADED) {
                return false
            }
        }
        return true
    }

    fun isFailed(): Boolean {
        for (file in getRecordingFiles()) {
            if (getStatusOfFileOrDir(file) == UploadStatus.FAILED) {
                return true
            }
        }
        return false
    }

    private fun statusEmoji(status: UploadStatus): String {
        return when (status) {
            UploadStatus.NOT_UPLOADED -> "\uD83D\uDCC5"
            UploadStatus.UPLOADED -> "✔️"
            UploadStatus.FAILED -> "❌"
            UploadStatus.UPLOADING -> "⏫"
            UploadStatus.UNKNOWN -> "❓"
        }
    }

    private fun statusLabel(status: UploadStatus): String {
        val errorString = if (error != null) error!!.message else ""
        return "${statusEmoji(status)} " + when (status) {
            UploadStatus.NOT_UPLOADED -> "Not uploaded yet"
            UploadStatus.UPLOADED -> "Uploaded"
            UploadStatus.FAILED -> "Upload failed: $errorString"
            UploadStatus.UPLOADING -> "Uploading"
            UploadStatus.UNKNOWN -> "Unknown"
        }
    }
}
