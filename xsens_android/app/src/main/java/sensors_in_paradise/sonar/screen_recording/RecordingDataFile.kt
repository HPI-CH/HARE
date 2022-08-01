package sensors_in_paradise.sonar.screen_recording

import com.opencsv.CSVReaderHeaderAware
import sensors_in_paradise.sonar.machine_learning.InMemoryWindow
import java.io.File
import java.io.FileReader
import java.io.IOException

class RecordingDataFile @Throws(IOException::class) constructor(private val mergedSensorDataFile: File) {
    private val indexesOfActivityChanges = findIndexesOfActivityChanges()

    class WindowException(msg: String) : Exception(msg)

    @Throws(WindowException::class)
    fun getWindowAtIndex(
        startingIndex: Int,
        windowSize: Int,
        featuresWithSensorTagPrefix: Array<String>
    ): Pair<InMemoryWindow, String> {
        val csvReader = getResetCsvReader()
        csvReader.skip(startingIndex)
        val window = InMemoryWindow(featuresWithSensorTagPrefix, windowSize)
        var activity: String? = null
        val errorMsgPrefix = "Window can't be filled from start line index $startingIndex: "
        for (i in 0 until (windowSize)) {
            val line: Map<String, String> = optNextLine(csvReader)
                ?: throw WindowException("$errorMsgPrefix Line is null before reaching window_size $windowSize")
            val lineActivity = line["activity"]
                ?: throw WindowException("$errorMsgPrefix Line without activity detected")
            val stf = line["SampleTimeFine"]?.replace(" ", "")?.toLong()
                ?: throw WindowException("$errorMsgPrefix Can't infer SampleTimeFine from line")

            if (activity == null) {
                activity = lineActivity
            }
            if (activity != lineActivity) {
                throw WindowException("$errorMsgPrefix Multiple activities " +
                        "within window have been detected: $lineActivity, $activity")
            }
            for ((feature, valueStr) in line) {
                if (window.needsFeature(feature)) {
                    val cleanedValueStr = valueStr.replace(" ", "")
                    val value = if (cleanedValueStr.isNotEmpty()) cleanedValueStr.toFloat() else 0f
                    window.appendSensorData(feature, value, stf)
                }
            }
        }
        return Pair(window, activity!!)
    }

    private fun getResetCsvReader(): CSVReaderHeaderAware {
        return CSVReaderHeaderAware(FileReader(mergedSensorDataFile))
    }

    @Suppress("LoopWithTooManyJumpStatements")
    fun getWindowStartIndexes(
        windowSize: Int,
        filterForActivities: Collection<String>? = null,
    ): ArrayList<Int> {
        val indexes = ArrayList<Int>()

        for (i in 0 until (indexesOfActivityChanges.size - 1)) {
            val startIndex = indexesOfActivityChanges[i].first
            val endIndex = indexesOfActivityChanges[i + 1].first - 1
            val activity = indexesOfActivityChanges[i].second
            val activityLength = endIndex - startIndex
            if (activityLength < windowSize) {
                continue
            }
            if (filterForActivities?.contains(activity) == false) {
                continue
            }
            val numCompletelyFittingWindows = activityLength / windowSize
            for (j in 0 until (numCompletelyFittingWindows)) {
                val windowStartIndex = startIndex + j * windowSize
                indexes.add(windowStartIndex)
                // 50% overlap
                val overlappingWindowStartIndex = windowStartIndex + windowSize / 2
                if (overlappingWindowStartIndex + windowSize <= endIndex) {
                    indexes.add(overlappingWindowStartIndex)
                }
            }
        }
        return indexes
    }

    private fun findIndexesOfActivityChanges(): ArrayList<Pair<Int, String?>> {
        val indexesOfActivityChanges = ArrayList<Pair<Int, String?>>()
        val csvReader = getResetCsvReader()
        var lastActivity: String? = null
        var line: Map<String, String>? = optNextLine(csvReader)

        var i = 0
        while (line != null) {
            val activity = line["activity"]
            if (activity != lastActivity) {
                indexesOfActivityChanges.add(Pair(i, activity))
                lastActivity = activity
            }
            i++
            line = optNextLine(csvReader)
        }
        indexesOfActivityChanges.add(Pair(i - 1, null))
        return indexesOfActivityChanges
    }

    private fun optNextLine(csvReader: CSVReaderHeaderAware): Map<String, String>? {
        return csvReader.readMap()
    }
}
