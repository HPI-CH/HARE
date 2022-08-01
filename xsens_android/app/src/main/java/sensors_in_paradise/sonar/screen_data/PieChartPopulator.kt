package sensors_in_paradise.sonar.screen_data

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import com.github.mikephil.charting.animation.Easing
import com.github.mikephil.charting.charts.PieChart
import com.github.mikephil.charting.data.PieData
import com.github.mikephil.charting.data.PieDataSet
import com.github.mikephil.charting.data.PieEntry
import com.github.mikephil.charting.formatter.ValueFormatter
import com.github.mikephil.charting.utils.ColorTemplate
import com.github.mikephil.charting.utils.MPPointF
import sensors_in_paradise.sonar.GlobalValues

class PieChartPopulator(val context: Context, private val chart: PieChart) {
    private val availableColors = ArrayList<Int>().apply {
        for (c in ColorTemplate.PASTEL_COLORS.apply { shuffle() }) this.add(c)
        for (c in ColorTemplate.VORDIPLOM_COLORS) this.add(c)
        for (c in ColorTemplate.MATERIAL_COLORS) this.add(c)
        for (c in ColorTemplate.JOYFUL_COLORS) this.add(c)
        for (c in ColorTemplate.LIBERTY_COLORS) this.add(c)
    }
    private var currentMaxColorIndex = 0
    private val labelToColorMap = mutableMapOf<String, Int>()
    fun populateAndAnimateChart(data: Map<String, Long>) {
        populateDurationPieChart(
            data
        )
        chart.animateY(1400, Easing.EaseInOutQuad)
    }

    private fun populateDurationPieChart(labelledDurations: Map<String, Long>) {
        val entries = ArrayList<PieEntry>()
        for ((activity, duration) in improveData(labelledDurations)) {
            entries.add(
                PieEntry(
                    duration.toFloat(),
                    activity
                )
            )
            if (activity !in labelToColorMap) {
                labelToColorMap[activity] = availableColors[currentMaxColorIndex++]
                currentMaxColorIndex %= availableColors.size
            }
        }
        val dataSet = PieDataSet(entries, "Recordings")
        prepareDataset(dataSet)
        val data = PieData(dataSet)

        data.setValueFormatter(object : ValueFormatter() {
            override fun getFormattedValue(value: Float): String {
                return GlobalValues.getDurationAsString(value.toLong())
            }
        })
        data.setValueTextSize(8f)
        data.setValueTextColor(GlobalValues.getAndroidColorResource(context, android.R.attr.textColorPrimary))
        data.setValueTypeface(Typeface.DEFAULT)

        chart.legend.isEnabled = false
        chart.setHoleColor(Color.TRANSPARENT)
        chart.data = data
        chart.highlightValues(null)
        chart.invalidate()
        chart.animate()
    }

    private fun prepareDataset(dataSet: PieDataSet) {
        dataSet.setDrawIcons(false)
        dataSet.sliceSpace = 3f
        dataSet.iconsOffset = MPPointF(0f, 40f)
        dataSet.selectionShift = 5f
        val colors = ArrayList<Int>()
        for (i in 0 until dataSet.entryCount) {
            val entry = dataSet.getEntryForIndex(i)
            colors.add(labelToColorMap[entry.label] ?: Color.CYAN)
        }
        dataSet.colors = colors
    }

    private fun improveData(
        data: Map<String, Long>,
        smallItemPercentageThreshold: Float = SMALL_ITEM_PERCENTAGE_THRESHOLD
    ): Map<String, Long> {
        val result = mutableMapOf<String, Long>()
        var sum = 0L
        for ((_, duration) in data) {
            sum += duration
        }
        for ((activity, duration) in data) {
            val label = if (duration >= sum * smallItemPercentageThreshold) activity else "other"
            result[label] = duration + (result[label] ?: 0L)
        }
        return result
    }

    companion object {
        const val SMALL_ITEM_PERCENTAGE_THRESHOLD = 0.04f
    }
}
