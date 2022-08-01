package sensors_in_paradise.sonar.screen_train

import org.json.JSONArray
import org.json.JSONObject
import sensors_in_paradise.sonar.JSONStorage
import sensors_in_paradise.sonar.use_cases.UseCase
import kotlin.math.round

class PredictionHistoryStorage(
    private val useCase: UseCase,
    private val startTimestamp: Long,
    private val shouldStorePrediction: Boolean = false
) {
    data class Prediction(var label: String, var percentage: Float) {
        fun percentageAsString(): String {
            val roundedPercentage = round(percentage * 100) / 100
            return "$roundedPercentage %"
        }
    }

    private var history: JSONArray
    val json = JSONObject().apply {
        put("startTimestamp", startTimestamp)
        history = JSONArray()
        put("history", history)
    }

    fun addPrediction(
        prediction: Prediction
    ): Long {
        val predictionObj = JSONObject()
        predictionObj.put("label", prediction.label)
        predictionObj.put("percentage", prediction.percentage)
        val relativeTime = System.currentTimeMillis() - startTimestamp
        predictionObj.put("relativeTimestamp", relativeTime)
        history.put(predictionObj)
        if (shouldStorePrediction) {
            JSONStorage.saveJSONObject(json, useCase.getPredictionHistoryJSONFile(startTimestamp))
        }
        return relativeTime
    }

    fun getPredictionHistory(sortLatestFirst: Boolean = true): ArrayList<Pair<Prediction, Long>> {
        val result = arrayListOf<Pair<Prediction, Long>>()
        for (i in 0 until history.length()) {
            val obj = history.getJSONObject(i)
            val label = obj.getString("label")
            val percentage = obj.getDouble("percentage").toFloat()
            val relativeTime = obj.getLong("relativeTimestamp")
            result.add(
                Pair(Prediction(label, percentage), relativeTime)
            )
        }
        if (sortLatestFirst) {
            result.sortByDescending { it.second }
        }
        return result
    }
}
