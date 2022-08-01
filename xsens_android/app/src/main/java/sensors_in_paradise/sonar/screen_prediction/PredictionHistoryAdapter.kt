package sensors_in_paradise.sonar.screen_prediction

import android.annotation.SuppressLint
import android.content.Context
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import org.apache.commons.lang3.tuple.MutablePair
import sensors_in_paradise.sonar.history.HistoryAdapter
import sensors_in_paradise.sonar.screen_train.PredictionHistoryStorage.Prediction
import java.util.concurrent.TimeUnit
import kotlin.collections.ArrayList
import kotlin.math.min
import kotlin.math.sqrt

class PredictionHistoryAdapter(
    val context: Context,
    predictionHistory: ArrayList<MutablePair<MutableList<Prediction>, Long>>
) : HistoryAdapter(true) {

    var predictionHistory = predictionHistory
        @SuppressLint("NotifyDataSetChanged")
        set(value) {
            field = value
            notifyDataSetChanged()
        }

    private fun formatTimestamp(millis: Long): String {
        val minutes = TimeUnit.MILLISECONDS.toMinutes(millis) % 60
        val minutesString = if (minutes < 10) " $minutes" else "$minutes"
        val seconds = TimeUnit.MILLISECONDS.toSeconds(millis) % 60
        val secondsString = if (seconds < 10) "0$seconds" else "$seconds"
        return "$minutesString:$secondsString"
    }

    private fun getAveragePercentage(
        predictions: List<Prediction>
    ): Float {
        return (predictions.sumOf { it.percentage.toDouble() } / predictions.size).toFloat()
    }

    fun addPrediction(
        prediction: Prediction,
        relativeTime: Long,
        recyclerView: RecyclerView? = null
    ) {
        val lastPrediction = predictionHistory.firstOrNull()?.left?.get(0)
        if (lastPrediction?.label == prediction.label) {
            predictionHistory[0].left.add(prediction)
            predictionHistory[0].right = relativeTime
            notifyItemChanged(0)
        } else {
            predictionHistory.add(0, MutablePair(mutableListOf(prediction), relativeTime))
            notifyItemAdded(0)
            recyclerView?.scrollToPosition(0)
        }
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        super.onBindViewHolder(holder, position)

        val paddingSize = min((sqrt(predictionHistory[position].left.size.toDouble() - 1) * 15f).toInt(), 200)
        holder.apply {
            val titleParam = titleTV.layoutParams as ViewGroup.MarginLayoutParams
            titleParam.setMargins(0, paddingSize, 0, 0)
            titleTV.layoutParams = titleParam
            val subTitleParam = subTitleTV.layoutParams as ViewGroup.MarginLayoutParams
            subTitleParam.setMargins(0, 0, 0, paddingSize)
            subTitleTV.layoutParams = subTitleParam
        }
    }

    override fun getPrefixOfItem(position: Int): String {
        val (_, timestamp) = predictionHistory[position]
        return formatTimestamp(timestamp)
    }

    override fun getTitleOfItem(position: Int): String {
        val (predictions, _) = predictionHistory[position]
        return predictions[0].label
    }

    override fun getSubtitleOfItem(position: Int): String {
        val (predictions, _) = predictionHistory[position]
        return if (predictions.size == 1) predictions[0].percentageAsString()
        else "Ø ${Prediction("", getAveragePercentage(predictions)).percentageAsString()}"
    }

    override fun getItemCount(): Int {
        return predictionHistory.size
    }
}
