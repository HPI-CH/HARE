package sensors_in_paradise.sonar.screen_data

import org.json.JSONArray
import org.json.JSONObject
import sensors_in_paradise.sonar.JSONStorage
import sensors_in_paradise.sonar.use_cases.UseCase

class DataHistoryStorage(useCase: UseCase) :
    JSONStorage(useCase.getTrainingHistoryJSONFile()) {
    data class TrainingOccasion(
        val timestamp: Long,
        val subdirectory: String,
        val peopleDurations: Map<String, Long>,
        val activityDurations: Map<String, Long>
    ) {
        fun getTotalDuration(): Long {
            var sum = 0L
            for ((_, duration) in peopleDurations) {
                sum += duration
            }
            return sum
        }

        fun getNumPeople(): Int {
            return peopleDurations.keys.size
        }

        fun getNumActivities(): Int {
            return activityDurations.keys.size
        }
    }

    private lateinit var history: JSONArray
    override fun onFileNewlyCreated() {
        json.put("history", JSONArray())
    }

    override fun onJSONInitialized() {
        history = json.getJSONArray("history")
    }

    fun addTrainingOccasion(
        subdirectory: String,
        trainingDataPeopleDistribution: Map<String, Long>,
        trainingDataActivityDistribution: Map<String, Long>
    ): TrainingOccasion {
        val occasionObj = JSONObject()
        val peopleDurationObj = storeMapInJsonObj(trainingDataPeopleDistribution)
        val activityDurationObj = storeMapInJsonObj(trainingDataActivityDistribution)
        occasionObj.put("peopleDurations", peopleDurationObj)
        occasionObj.put("activityDurations", activityDurationObj)
        occasionObj.put("subdirectory", subdirectory)
        val time = System.currentTimeMillis()
        occasionObj.put("timestamp", time)
        history.put(occasionObj)
        save()
        return TrainingOccasion(
            time,
            subdirectory,
            trainingDataPeopleDistribution,
            trainingDataActivityDistribution
        )
    }

    fun getTrainingHistory(sortLatestFirst: Boolean = true): ArrayList<TrainingOccasion> {
        val result = arrayListOf<TrainingOccasion>()
        for (i in 0 until history.length()) {
            val obj = history.getJSONObject(i)
            val subdirectory =
                obj.opt("subdirectory")?.toString() ?: UseCase.DEFAULT_RECORDINGS_SUB_DIR_NAME
            val timestamp = obj.getLong("timestamp")
            val activityDurations = getMapFromJsonObj(obj.getJSONObject("activityDurations"))
            val peopleDurations = getMapFromJsonObj(obj.getJSONObject("peopleDurations"))
            result.add(
                TrainingOccasion(
                    timestamp,
                    subdirectory,
                    peopleDurations,
                    activityDurations
                )
            )
        }
        if (sortLatestFirst) {
            result.sortByDescending { it.timestamp }
        }
        return result
    }

    private fun storeMapInJsonObj(map: Map<String, Long>): JSONObject {
        val result = JSONObject()
        for ((key, value) in map) {
            result.put(key, value)
        }
        return result
    }

    private fun getMapFromJsonObj(obj: JSONObject): Map<String, Long> {
        val result = mutableMapOf<String, Long>()
        val keys = obj.keys()
        for (key in keys) {
            result[key] = obj.getLong(key)
        }
        return result
    }
}
