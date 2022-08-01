package sensors_in_paradise.sonar.machine_learning

class Batch(
    private val dataSet: DataSet,
    private val windowStartIndexesPerRecording: ArrayList<ArrayList<Int>>
) {
    init {
        dataSet.addOnItemAddedListener { _, index ->
            windowStartIndexesPerRecording.add(index, ArrayList())
        }
        dataSet.addOnItemRemovedListener { _, index ->
            windowStartIndexesPerRecording.removeAt(index)
        }
    }

    /**Loads the specified batch data into RAM and return a Pair of windows and one hot encoded labels*/
    fun compile(
        model: TFLiteModel,
        acceptSmallerBatchesOnCompileException: Boolean = false,
        progressCallback: ((Int) -> Unit)? = null
    ): Pair<Array<Array<FloatArray>>, Array<FloatArray>> {
        val features = model.getFeaturesToPredict()
        val numFeatures = features.size
        val numWindows = windowStartIndexesPerRecording.sumOf { it.size }
        val windows = Array(numWindows) { Array(model.windowSize) { FloatArray(numFeatures) } }
        var windowIndex = 0
        val labelsOneHotEncoded = Array(numWindows) { FloatArray(model.getNumOutputClasses()) }
        for ((recordingIndex, startIndexes) in windowStartIndexesPerRecording.withIndex()) {
            for (startIndex in startIndexes) {
                val (window, label) = dataSet[recordingIndex].getWindowAtIndex(
                    startIndex,
                    model.windowSize,
                    features
                )
                try {
                    window.compileWindowToArray(windows[windowIndex])
                    labelsOneHotEncoded[windowIndex] = model.convertLabelToOneHotEncoding(label)
                } catch (e: IllegalArgumentException) {
                    if (!acceptSmallerBatchesOnCompileException) {
                        throw e
                    }
                }
                    windowIndex++
                progressCallback?.let { it((windowIndex * 100) / numWindows) }
            }
        }
        return Pair(windows, labelsOneHotEncoded)
    }
}
