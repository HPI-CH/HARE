package sensors_in_paradise.sonar.screen_prediction.barChart

import android.os.Handler
import android.view.animation.Interpolator
import android.view.animation.LinearInterpolator
import com.github.mikephil.charting.charts.BarLineChartBase
import com.github.mikephil.charting.charts.Chart
import com.github.mikephil.charting.data.BarDataSet
import com.github.mikephil.charting.data.BarEntry
import com.github.mikephil.charting.data.Entry
import java.util.*

class AnimateDataSetChanged(
    private val duration: Int,
    private val chart: Chart<*>,
    oldData: List<Entry>?,
    newData: List<Entry>?
) {
    private var startTime: Long = 0
    private var fps = 60
    private var timerHandler: Handler? = null
    private val oldData: List<Entry>
    private val newData: List<Entry>
    private var interpolator: Interpolator
    fun setInterpolator(interpolator: Interpolator) {
        this.interpolator = interpolator
    }

    fun run(fps: Int) {
        this.fps = fps
        run()
    }

    fun run() {
        startTime = Calendar.getInstance().timeInMillis
        timerHandler = Handler()
        val runner: Runner = Runner()
        runner.run()
    }

    private inner class Runner : Runnable {
        override fun run() {
            var increment = (Calendar.getInstance().timeInMillis - startTime) / duration.toFloat()
            increment =
                interpolator.getInterpolation(if (increment < 0f) 0f else if (increment > 1f) 1f else increment)
            chart.data.getDataSetByIndex(0).clear()
            for (i in newData.indices) {
                val oldY = if (oldData.size > i) oldData[i].y else newData[i].y
                val oldX = if (oldData.size > i) oldData[i].x else newData[i].x
                val newX = newData[i].x
                val newY = newData[i].y
                val e = BarEntry(oldX + (newX - oldX) * increment, oldY + (newY - oldY) * increment)
                (chart.data.getDataSetByIndex(0) as BarDataSet).addEntry(e)
            }
            chart.xAxis.resetAxisMaximum()
            chart.xAxis.resetAxisMinimum()
            chart.notifyDataSetChanged()
            chart.refreshDrawableState()
            chart.invalidate()
            if (chart is BarLineChartBase<*>) {
                chart.isAutoScaleMinMaxEnabled = true
            }
            if (increment < 1f) {
                timerHandler!!.postDelayed(this, (1000 / fps).toLong())
            }
        }
    }

    init {
        this.oldData = oldData?.let { ArrayList(it) }!!
        this.newData = ArrayList(newData)
        interpolator = LinearInterpolator()
    }
}
