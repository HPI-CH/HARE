package sensors_in_paradise.sonar.screen_prediction.barChart

import android.content.Context
import android.graphics.Typeface
import android.view.animation.LinearInterpolator
import com.github.mikephil.charting.charts.BarChart
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.BarData
import com.github.mikephil.charting.data.BarDataSet
import com.github.mikephil.charting.data.BarEntry
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter
import com.github.mikephil.charting.formatter.ValueFormatter
import sensors_in_paradise.sonar.R
import sensors_in_paradise.sonar.screen_train.PredictionHistoryStorage
import kotlin.math.min

class PredictionBarChart(
    private val context: Context,
    private val barChart: BarChart,
    numOutputs: Int,
    private val animationDurationMs: Long = 450L
) {
    private val numBars = min(numOutputs, 6)

    init {
        barChart.description.isEnabled = false
        barChart.legend.isEnabled = false

        barChart.setDrawBarShadow(false)
        barChart.setDrawValueAboveBar(true)
        barChart.setTouchEnabled(false)
        barChart.setDrawGridBackground(false)

        barChart.setFitBars(false)
        barChart.axisLeft.isEnabled = false
        barChart.axisRight.isEnabled = false

        val range = (0 until numBars)

        barChart.xAxis.position = XAxis.XAxisPosition.BOTTOM
        barChart.xAxis.setDrawAxisLine(false)
        barChart.xAxis.setDrawGridLines(false)
        barChart.xAxis.textColor = context.getColor(R.color.hardBackgroundContrast)
        barChart.xAxis.typeface = Typeface.DEFAULT_BOLD

        val initValues = range.map { i ->
            BarEntry(
                i.toFloat(),
                0.5f
            )
        }
        val dataSet = BarDataSet(initValues, "")
        dataSet.setValueTextColors((range).map { context.getColor(R.color.hardBackgroundContrast) })
        dataSet.valueFormatter = object : ValueFormatter() {
            override fun getFormattedValue(value: Float): String {
                return "${value.toInt()} %"
            }
        }

        val barData = BarData(dataSet)
        barData.setValueTextSize(10f)
        barData.setValueTypeface(Typeface.DEFAULT_BOLD)
        barData.barWidth = 0.9f

        barChart.data = barData
        resetData()
    }

    private fun getLimitedPredictions(
        predictions: ArrayList<PredictionHistoryStorage.Prediction>
    ): ArrayList<PredictionHistoryStorage.Prediction> {
        return if (predictions.size <= numBars) {
            predictions
        } else {
            val limitedPredictions = ArrayList<PredictionHistoryStorage.Prediction>()
            var othersPercentage = 0f
            for ((i, prediction) in predictions.withIndex()) {
                if (i < numBars - 1) {
                    limitedPredictions.add(prediction)
                } else {
                    othersPercentage += prediction.percentage
                }
            }
            limitedPredictions.add(PredictionHistoryStorage.Prediction("Others", othersPercentage))
            limitedPredictions
        }
    }

    fun resetData() {
        changeData(
            ArrayList((0 until numBars).map { PredictionHistoryStorage.Prediction("", 0.5f) }),
            "_"
        )
    }

    fun changeData(
        predictions: ArrayList<PredictionHistoryStorage.Prediction>,
        highestPrediction: String
    ) {
        val limitedPredictions = getLimitedPredictions(predictions)
        val oldValues = (0 until numBars).map { barChart.data.dataSets[0].getEntryForIndex(it) }
        val newValues = limitedPredictions.mapIndexed { i, prediction ->
            BarEntry(
                i.toFloat(),
                prediction.percentage
            )
        }

        val xAxisLabels = limitedPredictions.map { p ->
            if (p.label.length <= 10) p.label
            else "${p.label.substring(0, 9)}.."
        }
        barChart.xAxis.valueFormatter = IndexAxisValueFormatter(xAxisLabels)

        val dataSet = barChart.data.dataSets[0] as BarDataSet
        dataSet.colors = (limitedPredictions).map {
            if (it.label == highestPrediction) context.getColor(R.color.colorPrimaryDark)
            else context.getColor(R.color.colorAccent)
        }

        val changer = AnimateDataSetChanged(
            animationDurationMs.toInt(),
            barChart,
            oldValues,
            newValues
        )
        changer.setInterpolator(LinearInterpolator())
        changer.run()
    }
}
