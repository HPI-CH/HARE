package sensors_in_paradise.sonar.custom_views.confusion_matrix

import android.util.Log

open class ConfusionMatrix(
    labels: Array<String>,
    var title: String = "Confusion Matrix",
    var description: String? = null
) {
    constructor(confusionMatrix: ConfusionMatrix) : this(
        confusionMatrix.getLabels().toTypedArray(),
        confusionMatrix.title
    ) {
        for (col in 0 until getNumLabels()) {
            for (row in 0 until getNumLabels()) {
                this[col, row] = confusionMatrix[col, row]
            }
        }
        this.maxCellValue = confusionMatrix.maxCellValue
    }

    val labels = labels.mapIndexed { index, label ->
        label to index
    }.toMap()
    private val data = ArrayList<ArrayList<Int>>()
    private var maxCellValue = 0

    init {
        initData()
    }

    private fun initData() {
        data.clear()
        repeat(getNumLabels()) {
            val column = ArrayList<Int>()
            repeat(getNumLabels()) {
                column.add(0)
            }
            data.add(column)
        }
    }

    operator fun get(column: Int, row: Int): Int {
        return data[column][row]
    }

    operator fun set(column: Int, row: Int, value: Int) {
        data[column][row] = value
    }

    fun getColumn(column: Int): Collection<Int> {
        return data[column]
    }

    fun getNumLabels(): Int {
        return labels.size
    }

    fun getLabels(): Collection<String> {
        return labels.keys
    }

    fun getMaxCellValue(): Int {
        return maxCellValue
    }

    fun addPredictions(predictions: Array<String>, actualLabels: Array<String>) {
        if (predictions.size != actualLabels.size) {
            throw IllegalArgumentException(
                "Size of predictions (${predictions.size}) must " +
                        "be equal to size of labels (${actualLabels.size})"
            )
        }
        for (i in predictions.indices) {
            val prediction = predictions[i]
            val label = actualLabels[i]

            val predictionIndex = labels[prediction]
            val labelIndex = labels[label]

            if (predictionIndex == null || labelIndex == null) {
                throw IllegalArgumentException(
                    "Prediction ($prediction) or actual label " +
                            "($label) at index $i is not in specified labels this ConfusionMatrix " +
                            "was initialized with."
                )
            }
            Log.d(
                "ConfusionMatrix",
                "Current maxCellValue: $maxCellValue, value of current cell: ${this[predictionIndex, labelIndex] + 1}"
            )
            if (++this[predictionIndex, labelIndex] > maxCellValue) {
                maxCellValue = this[predictionIndex, labelIndex]
            }
        }
    }

    override fun toString(): String {
        var result = ""

        for (col in 0 until getNumLabels()) {
            for (row in 0 until getNumLabels()) {
                result += this[row, col].toString() + " "
            }
            result += "\n"
        }
        return result
    }

    companion object {
        const val COLUMN_AXIS_LABEL = "Predicted label"
        const val ROW_AXIS_LABEL = "Actual label"
    }
}
