package sensors_in_paradise.sonar.screen_data

import android.annotation.SuppressLint
import sensors_in_paradise.sonar.GlobalValues
import sensors_in_paradise.sonar.history.HistoryAdapter
import java.text.DateFormat
import java.util.*
import kotlin.collections.ArrayList

class DataHistoryAdapter(trainingHistory: ArrayList<DataHistoryStorage.TrainingOccasion>) :
    HistoryAdapter() {
    var trainingHistory = trainingHistory
        @SuppressLint("NotifyDataSetChanged")
        set(value) {
            field = value
            notifyDataSetChanged()
        }
    private val dateFormat = DateFormat.getDateTimeInstance()

    override fun getTitleOfItem(position: Int): String {
        val occasion = trainingHistory[position]
        return dateFormat.format(Date(occasion.timestamp))
    }

    override fun getSubtitleOfItem(position: Int): String {
        val occasion = trainingHistory[position]
        var subtitle = "Duration ${GlobalValues.getDurationAsString(occasion.getTotalDuration())}"
        subtitle += "\n\uD83D\uDCC2 ${occasion.subdirectory}"
        subtitle += "\n${occasion.getNumActivities()} activities"
        subtitle += "\n${occasion.getNumPeople()} people"

        return subtitle
    }

    override fun getItemCount(): Int {
        return trainingHistory.size
    }
}
