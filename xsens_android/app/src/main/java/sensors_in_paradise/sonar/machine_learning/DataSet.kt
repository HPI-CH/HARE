package sensors_in_paradise.sonar.machine_learning

import sensors_in_paradise.sonar.ObservableArrayList
import sensors_in_paradise.sonar.screen_recording.RecordingDataFile
import kotlin.math.ceil

open class DataSet : ObservableArrayList<RecordingDataFile>() {

    fun convertToBatches(
        windowsPerBatch: Int,
        windowSize: Int,
        shuffle: Boolean = true,
        filterForActivities: Collection<String>? = null,
        progressCallback: ((Int) -> Unit)? = null
    ): ArrayList<Batch> {
        val windowStartIndexesPerRecording = this.map { it.getWindowStartIndexes(windowSize, filterForActivities) }

        val indexesOfIndexes = ArrayList<Pair<Int, Int>>()
        for ((recordingIndex, startIndexes) in windowStartIndexesPerRecording.withIndex()) {
            for (startIndex in startIndexes) {
                indexesOfIndexes.add(Pair(recordingIndex, startIndex))
            }
        }
        if (shuffle) {
            indexesOfIndexes.shuffle()
        }
        val numWindows = indexesOfIndexes.size
        val numBatches = numWindows / windowsPerBatch
        val result = ArrayList<Batch>(numBatches)
        for (b in 0 until numBatches) {
            val batchStartIndexesPerRecording = ArrayList(this.map { ArrayList<Int>() })
            for (i in b * windowsPerBatch until b * windowsPerBatch + windowsPerBatch) {
                val (recordingIndex, windowStartIndex) = indexesOfIndexes[i]
                batchStartIndexesPerRecording[recordingIndex].add(windowStartIndex)
            }
            result.add(Batch(this, batchStartIndexesPerRecording))
            progressCallback?.let { it((b * 100) / numBatches) }
        }
        return result
    }

    /**Splits the current DataSet into 2 DataSets and returns them as a pair.
     * Tries to make the relative size of the second data set close to the parameter `secondDataSetPercentage`.
     * However it will at least put 1 Recording into each data set.
     * */
    fun splitByPercentage(secondDataSetPercentage: Float = 0.2f): Pair<DataSet, DataSet> {
        if (size < 2) {
            throw IllegalArgumentException("Can't split a dataset of size $size into two parts.")
        }
        val secondDataSetSize = ceil(secondDataSetPercentage * size).toInt()
        val indices = (0 until size).shuffled()
        val second = DataSet()
        val first = DataSet()

        for ((entryIndex, recordingIndex) in indices.withIndex()) {
            (if (entryIndex < secondDataSetSize) second else first).add(this[recordingIndex])
        }
        return Pair(first, second)
    }

    fun convertToTrainValBatches(
        windowsPerBatch: Int,
        windowSize: Int,
        splitPercentage: Float,
        filterForActivities: Collection<String>? = null,
        progressCallback: ((Int) -> Unit)? = null
    ): Pair<ArrayList<Batch>, ArrayList<Batch>> {
        val batches = convertToBatches(
            windowsPerBatch = windowsPerBatch,
            windowSize = windowSize,
            shuffle = true,
            filterForActivities = filterForActivities,
            progressCallback = progressCallback
        )
        val splitIndex = ceil(splitPercentage * batches.size).toInt()
        val train = ArrayList(batches.subList(0, splitIndex))
        val validation = ArrayList(batches.subList(splitIndex, batches.size))
        return Pair(train, validation)
    }
}
