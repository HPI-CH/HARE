package sensors_in_paradise.sonar.screen_recording

import android.util.Log
import sensors_in_paradise.sonar.GlobalValues
import sensors_in_paradise.sonar.ObservableArrayList
import java.io.File

class RecordingDataManager(recordingsDir: File) :
    ObservableArrayList<Recording>() {
    var recordingsDir: File = recordingsDir
        set(value) {
            field = value
            loadRecordingsFromStorage()
        }

    init {
        loadRecordingsFromStorage()
    }

    fun reloadRecordingsFromStorage() {
        loadRecordingsFromStorage()
    }

    private fun loadRecordingsFromStorage() {
        clear()

        recordingsDir.walk().forEach {
            if (it.isDirectory) {
                if (isRecordingDir(it)) {
                    add(Recording(it))
                }
            }
        }
        sortByDescending { recording -> recording.metadataStorage.getTimeStarted() }
    }

    private fun isRecordingDir(file: File): Boolean {
        val childDirs = file.listFiles { dir, filename -> dir.resolve(filename).isDirectory }
        if (childDirs == null || childDirs.isEmpty()) {
            val activeFlagFile = file.resolve(GlobalValues.ACTIVE_RECORDING_FLAG_FILENAME)
            val metadataFile = file.resolve(GlobalValues.METADATA_JSON_FILENAME)
            return metadataFile.exists() && !activeFlagFile.exists()
        }
        return false
    }

    fun getNumberOfRecordingsPerActivity(): Map<String, Int> {
        val activities = ArrayList<String>()
        for (rec in this) {
            val storage = rec.metadataStorage
            activities.addAll(storage.getActivities().map { (_, label) -> label })
        }
        return activities.groupingBy { it }.eachCount()
    }

    fun deleteRecording(recording: Recording) {
        recording.delete()
        remove(recording)
    }

    fun getActivityDurations(
        filterForPerson: String? = null,
        onlyUntrainedRecordings: Boolean = false
    ): Map<String, Long> {
        return getActivityDurations(this, filterForPerson, onlyUntrainedRecordings)
    }

    fun getPeopleDurations(
        filterForActivity: String? = null,
        onlyUntrainedRecordings: Boolean = false
    ): Map<String, Long> {
        return Companion.getPeopleDurations(this, filterForActivity, onlyUntrainedRecordings)
    }

    fun getRecordingsBySubject(
        subject: String,
        includeAlreadyTrainedOnRecordings: Boolean
    ): List<Recording> {
        return this.filter {
            it.metadataStorage.getPerson() == subject &&
                    (includeAlreadyTrainedOnRecordings || (!it.metadataStorage.hasBeenUsedForOnDeviceTraining()))
        }
    }

    companion object {

        fun getActivityDurations(
            recordings: List<Recording>,
            filterForPerson: String? = null,
            onlyUntrainedRecordings: Boolean = false
        ): Map<String, Long> {
            val result = mutableMapOf<String, Long>()

            val filteredRecordings =
                recordings.filter { !it.metadataStorage.hasBeenUsedForOnDeviceTraining() || !onlyUntrainedRecordings }
                    .filter { it.metadataStorage.getPerson() == filterForPerson || filterForPerson == null }
            for (recording in filteredRecordings) {
                val activityDuration = getActivityDurations(recording)
                activityDuration.forEach { (name, duration) ->
                    result.merge(
                        name,
                        duration,
                        Long::plus
                    )
                }
            }
            return result
        }

        private fun getActivityDurations(
            recording: Recording
        ): Map<String, Long> {
            val result = mutableMapOf<String, Long>()
            val metadata = recording.metadataStorage

            val activities = metadata.getActivities()
            for ((index, labelEntry) in activities.withIndex()) {
                val duration = RecordingMetadataStorage.getDurationOfActivity(
                    activities,
                    index,
                    metadata.getTimeEnded()
                )
                result[labelEntry.activity] = duration + (result[labelEntry.activity] ?: 0L)
            }

            return result
        }

        fun getPeopleDurations(
            recordings: List<Recording>,
            filterForActivity: String? = null,
            onlyUntrainedRecordings: Boolean = false
        ): Map<String, Long> {
            val result = mutableMapOf<String, Long>()
            for (recording in recordings.filter {
                !onlyUntrainedRecordings || !it.metadataStorage.hasBeenUsedForOnDeviceTraining()
            }) {
                val peopleDuration = getPeopleDurations(filterForActivity, recording)
                peopleDuration.forEach { (name, duration) ->
                    result.merge(
                        name,
                        duration,
                        Long::plus
                    )
                }
            }
            return result
        }

        private fun getPeopleDurations(
            filterForActivity: String? = null,
            recording: Recording
        ): Map<String, Long> {
            val result = mutableMapOf<String, Long>()
            val metadata = recording.metadataStorage
            val person = metadata.getPerson()
            if (filterForActivity != null) {
                val activities = metadata.getActivities()
                for ((index, labelEntry) in activities.withIndex()) {
                    val (_, activity) = labelEntry
                    if (activity == filterForActivity) {
                        val duration = RecordingMetadataStorage.getDurationOfActivity(
                            activities,
                            index,
                            metadata.getTimeEnded()
                        )
                        result[person] = duration + (result[person] ?: 0L)
                    }
                }
            } else {
                result[person] = metadata.getDuration() + (result[person] ?: 0L)
            }
            return result
        }

        @Throws(Recording.InvalidRecordingException::class)
        @Suppress("TooGenericExceptionCaught", "SwallowedException")
        fun convertRecordings(
            recordings: List<Recording>,
            skipInvalidRecordings: Boolean = true,
            regenerateExistingFiles: Boolean = false,
            callback: (progress: Int) -> Unit
        ): List<RecordingDataFile> {
            val result = ArrayList<RecordingDataFile>(recordings.size)
            for ((index, recording) in recordings.withIndex()) {
                try {
                    val shouldGenerateMergedFile =
                        recording.hasMergedSensorFile() && !regenerateExistingFiles
                    val dataFile =
                        if (shouldGenerateMergedFile) recording.getMergedSensorFile() else recording.mergeSensorFiles()
                    result.add(RecordingDataFile(dataFile))
                } catch (e: Exception) {
                    if (!skipInvalidRecordings) {
                        throw Recording.InvalidRecordingException(
                            e.message ?: "Could not convert ${recording.dir}"
                        )
                    }
                    Log.w(
                        "RecordingDataManager-convertRecordings",
                        "Exception occured when converting recording: ${e.message}"
                    )
                }
                callback(((index + 1) * 100) / recordings.size)
            }
            return result
        }
    }
}
