package sensors_in_paradise.sonar.screen_recording

import android.util.Log
import com.opencsv.CSVReaderHeaderAware
import sensors_in_paradise.sonar.GlobalValues
import sensors_in_paradise.sonar.XSensDotDeviceWithOfflineMetadata
import java.io.*
import kotlin.math.floor
import kotlin.random.Random

const val XSENS_HEADER_SIZE = 9
const val XSENS_EMPTY_FILE_SIZE = 435

/**
 * Enum that describes the condition / state a recording can be in
 */
enum class RecordingFileState {
    Empty,
    Unsynchronized,
    Valid,
    WithoutSensor
}

open class Recording(val dir: File, var metadataStorage: RecordingMetadataStorage) {
    constructor(dir: File) : this(
        dir,
        RecordingMetadataStorage(dir.resolve(GlobalValues.METADATA_JSON_FILENAME))
    )

    constructor(recording: Recording) : this(
        recording.dir,
        recording.metadataStorage
    )

    val state = computeRecordingState()
    val isValid
        get() = state != RecordingFileState.Empty

    private fun doSensorFilesExist(dir: File): Boolean {
        var childCSVs = dir.listFiles { _, name -> name.endsWith(".csv") }?.toList()
        childCSVs = childCSVs?.filterNot { csvFile -> csvFile.name == POSE_CAPTURE_FILENAME }
        if (childCSVs == null || childCSVs.isEmpty()) {
            return true
        }
        return false
    }

    private fun areFilesEmpty(dir: File): Boolean {
        var childCSVs = dir.listFiles { _, name -> name.endsWith(".csv") }?.toList()
        childCSVs = childCSVs?.filterNot { csvFile -> csvFile.name == POSE_CAPTURE_FILENAME }
        if (childCSVs != null && childCSVs.isNotEmpty()) {
            for (child in childCSVs) {
                if (child.length() < XSENS_EMPTY_FILE_SIZE) {
                    return true
                }
            }
            return false
        }
        return true
    }

    fun getDisplayTitle(): String {
        val numActivities = metadataStorage.getActivities().size
        var result = ""
        if (hasVideoRecording()) {
            result += "\uD83D\uDCF9 "
        }
        if (hasPoseSequenceRecording()) {
            result += "\uD83E\uDD38 "
        }
        result += "$numActivities ${if (numActivities == 1) "activity" else "activities"}"
        return result
    }

    fun getDirectory(): File {
        return dir
    }

    fun delete() {
        val children = dir.listFiles()
        if (children != null) {
            for (child in children) {
                child.delete()
            }
        }
        dir.delete()
    }

    fun hasVideoRecording(): Boolean {
        return getVideoFile().exists()
    }

    fun getVideoFile(): File {
        return dir.resolve(VIDEO_CAPTURE_FILENAME)
    }

    fun hasPoseSequenceRecording(): Boolean {
        return getPoseSequenceFile().exists()
    }

    fun getPoseSequenceFile(): File {
        return dir.resolve(POSE_CAPTURE_FILENAME)
    }

    private fun computeRecordingState(): RecordingFileState {
        var state = checkCache()
        if (state != null) return state

        state = if (doSensorFilesExist(dir)) {
            RecordingFileState.WithoutSensor
        } else if (areFilesEmpty(dir)) {
            RecordingFileState.Empty
        } else if (!areFilesSynchronized()) {
            RecordingFileState.Unsynchronized
        } else {
            RecordingFileState.Valid
        }

        saveCache(state)
        return state
    }

    /**
     * Checks if the files of the recording are synchronized -> contain exactly the same timestamps.
     * **This only works if the files are not empty.**
     *
     * This is checked on one sample line, as the timestamps are usually consistently spaced for one
     * sensor.
     */
    private fun areFilesSynchronized(): Boolean {
        val childCSVs = dir.listFiles { _, name -> name.endsWith(".csv") } ?: return false

        val firstFile = childCSVs[0]

        val lineNumber = getAbsoluteLineNumber(firstFile)
        val timesteps = lineNumber - XSENS_HEADER_SIZE

        val margin = floor(timesteps * 0.2).toInt()
        val lineFrom = XSENS_HEADER_SIZE + margin
        val lineTo = lineNumber - margin
        // End is exclusive, so +1 on last line
        val randomLine = Random.nextInt(lineFrom, lineTo + 1)
        val timestamp = getTimeStampAtLine(firstFile, randomLine)

        assert(timestamp != "") { "No initial timestamp could be found." }

        for (child in childCSVs) {
            if (!findTimeStamp(child, timestamp)) {
                return false
            }
        }

        return true
    }

    private fun getTimeStampAtLine(file: File, lineNumber: Int): String {
        var counter = 0

        val reader = BufferedReader(FileReader(file))
        var line: String
        while (true) {
            line = reader.readLine() ?: break

            if (counter == lineNumber) {
                val columns = line.split(",")
                reader.close()
                return columns[1]
            }

            counter++
        }

        reader.close()
        return ""
    }

    private fun findTimeStamp(file: File, timestamp: String): Boolean {
        var headerSize = 9

        val reader = BufferedReader(FileReader(file))
        var line: String
        while (true) {
            line = reader.readLine() ?: break

            if (headerSize <= 0) {
                val columns = line.split(",")
                if (columns[1] == timestamp) {
                    reader.close()
                    return true
                }
            } else {
                headerSize--
            }
        }

        reader.close()
        return false
    }

    /**
     * Counts all lines in the given recording file **including** the header
     */
    private fun getAbsoluteLineNumber(file: File): Int {
        var lineNumber = 0

        val reader = BufferedReader(FileReader(file))
        while (reader.readLine() != null) lineNumber++
        reader.close()

        return lineNumber
    }

    fun getRecordingFiles(): Array<File> {
        return dir.listFiles { file -> file.isFile && file.name.endsWith(".csv") &&
                file.name != MERGED_SENSOR_DATA_FILE_NAME }
            ?: emptyArray()
    }

    private fun checkCache(): RecordingFileState? {
        val state = metadataStorage.getRecordingState()
        if (state == null) return state

        return when (state) {
            "Without Sensors" -> {
                RecordingFileState.WithoutSensor
            }
            "Valid" -> {
                RecordingFileState.Valid
            }
            "Empty" -> {
                RecordingFileState.Empty
            }
            else -> {
                RecordingFileState.Unsynchronized
            }
        }
    }

    private fun saveCache(state: RecordingFileState) {
        val recordingState = when (state) {
            RecordingFileState.WithoutSensor -> {
                "Without Sensors"
            }
            RecordingFileState.Valid -> {
                "Valid"
            }
            RecordingFileState.Empty -> {
                "Empty"
            }
            else -> {
                "Unsynchronized"
            }
        }

        metadataStorage.setRecordingState(recordingState)
    }

    fun getActivitiesSummary(): String {
        return metadataStorage.getActivities().joinToString("\n") { (activityStartTime, activity) ->
            GlobalValues.getDurationAsString(activityStartTime - metadataStorage.getTimeStarted()) + "   " +
                    activity
        }
    }

    private fun getSensorTagPrefixForRecordingFile(
        recordingFile: File,
        sensorMacMap: Map<String, String>? = null
    ): String? {
        val sensorAddress = recordingFile.name.substringBeforeLast(".")

        val addressToTag = sensorMacMap ?: metadataStorage.getSensorMacMap()
        val sensorTag = addressToTag[sensorAddress] ?: addressToTag[sensorAddress.replace("-", ":")]
        ?: return null
        return XSensDotDeviceWithOfflineMetadata.extractTagPrefixFromTag(sensorTag)
    }

    /** Returns a map of sensor tag prefix to CSVReader
     */
    private fun getCSVReadersOfSensorRecordings(): LinkedHashMap<String, CSVReaderHeaderAware> {
        val recordingSensorTagToReaderMap = LinkedHashMap<String, CSVReaderHeaderAware>()
        val sensorMacMap = metadataStorage.getSensorMacMap()
        for (file in getRecordingFiles()) {
            val fileReader = GlobalValues.getCSVHeaderAwareFileReader(file)
            val csvReader = CSVReaderHeaderAware(fileReader)

            val sensorAddress = file.name.substringBeforeLast(".")
            val sensorTagPrefix = getSensorTagPrefixForRecordingFile(file, sensorMacMap)
            if (sensorTagPrefix == null) {
                Log.w(
                    "Recording-mergeSensorFiles",
                    "Can't include sensor data file ${file.name} of recording ${dir.name} " +
                            "in merge: Could not infer sensor tag from address $sensorAddress"
                )
                continue
            }
            recordingSensorTagToReaderMap[sensorTagPrefix] = csvReader
        }
        return recordingSensorTagToReaderMap
    }

    fun hasMergedSensorFile(): Boolean {
        return getMergedSensorFile().exists()
    }

    fun getMergedSensorFile(): File {
        return File(dir, MERGED_SENSOR_DATA_FILE_NAME)
    }

    @Throws(InvalidRecordingException::class)
    @Suppress("LongMethod", "ComplexMethod")
    fun mergeSensorFiles(): File {
        val outFile = getMergedSensorFile()
        val recordingSensorTagToReaderMap = getCSVReadersOfSensorRecordings()

        val resultLineEntries = LinkedHashMap<String, String>()
        val mergedCSVWriter = PrintWriter(FileWriter(outFile))
        var hasWrittenColumnNames = false
        var hasNewContent = true

        var startSampleTimeFine: Long? = null
        do {
            if (recordingSensorTagToReaderMap.isEmpty()) {
                break
            }
            try {
                var sampleTimeFine: Long? = null
                for ((tagPrefix, csvReader) in recordingSensorTagToReaderMap) {
                    val line = csvReader.readMap()
                    if (line == null) {
                        hasNewContent = false
                        break
                    }
                    for ((key, value) in line.filter { it.key !in sensorDataColumnsToIgnoreInMerge }) {
                        // since the linked hash map is defined outside of the for loop,
                        // we preserve the order of its keys
                        resultLineEntries["${key}_$tagPrefix"] = value
                    }
                    if (sampleTimeFine == null) {
                        val stf = line["SampleTimeFine"]
                            ?: throw InvalidRecordingException(
                                "Could not find SampleTimeFine " +
                                        "in sensor file of sensor with tag $tagPrefix "
                            )
                        sampleTimeFine = stf.replace(" ", "").toLong()
                        resultLineEntries["SampleTimeFine"] = stf
                    }
                    if (startSampleTimeFine == null) {
                        startSampleTimeFine = sampleTimeFine
                    }
                }
                if (hasNewContent) {
                    val timeMsIntoRecording = (sampleTimeFine!! - startSampleTimeFine!!) / 1000L
                    val activity = metadataStorage.getActivityAtTime(timeMsIntoRecording)
                        ?: throw InvalidRecordingException(
                            "Could not find activity for current " +
                                    "row ($timeMsIntoRecording ms into recording)"
                        )

                    resultLineEntries["activity"] = activity
                    if (!hasWrittenColumnNames) {
                        mergedCSVWriter.println(resultLineEntries.keys.joinToString(","))
                        hasWrittenColumnNames = true
                    }
                    mergedCSVWriter.println(resultLineEntries.values.joinToString(","))
                }
            } catch (_: Exception) {
                hasNewContent = false
            }
        } while (hasNewContent)
        try {
            mergedCSVWriter.close()
            for ((_, csvReader) in recordingSensorTagToReaderMap) {
                csvReader.close()
            }
        } catch (e: IOException) {
            Log.w(
                "Recording-mergeSensorFiles",
                "IOException while closing csv reader or writer ${e.message}"
            )
        }
        return outFile
    }

    class InvalidRecordingException(msg: String) : Exception(msg)

    companion object {
        const val VIDEO_CAPTURE_FILENAME = "recording.mp4"
        const val POSE_CAPTURE_FILENAME = "poseSequence.csv"
        private const val MERGED_SENSOR_DATA_FILE_NAME = "allSensorData.csv"
        private val sensorDataColumnsToIgnoreInMerge =
            arrayOf("PacketCounter", "Status", "SampleTimeFine")
    }
}
