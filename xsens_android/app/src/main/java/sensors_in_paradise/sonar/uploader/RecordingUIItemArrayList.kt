package sensors_in_paradise.sonar.uploader

import java.io.File

class RecordingUIItemArrayList : ArrayList<RecordingUIItem>() {

    fun getRecordingsInDir(dir: File): ArrayList<RecordingUIItem> {
        val recordings = ArrayList<RecordingUIItem>()
        for (fileUIItem in this) {
            if (fileUIItem.dir.path.contains(dir.path)) {
                recordings.add(fileUIItem)
            }
        }
        return recordings
    }

	fun getByRecordingDir(dir: File): RecordingUIItem? {
        for (fileUIItem in this) {
            if (fileUIItem.dir == dir) {
                return fileUIItem
            }
        }
        return null
    }

	fun areAllUploadedOrFailed(): Boolean {
        for (recording in this) {
            if (!recording.isUploaded() && !recording.isFailed()) {
                return false
            }
        }
        return true
    }

	fun areAllUploaded(): Boolean {
        for (item in this) {
            if (!item.isUploaded()) {
                return false
            }
        }
        return true
    }
}
