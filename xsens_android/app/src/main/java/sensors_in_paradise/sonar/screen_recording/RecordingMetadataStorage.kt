@file:Suppress("SwallowedException")

package sensors_in_paradise.sonar.screen_recording

import android.os.Build
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import sensors_in_paradise.sonar.JSONStorage
import java.io.File
import java.lang.IndexOutOfBoundsException

class RecordingMetadataStorage(file: File, initialJson: JSONObject? = null) :
    JSONStorage(file, initialJson) {
    data class LabelEntry(
        var timeStarted: Long,
        var activity: String
    )

    private lateinit var activities: JSONArray
    override fun onFileNewlyCreated() {
        activities = JSONArray()
        json.put("activities", activities)
        save()
    }

    override fun onJSONInitialized() {
        activities = json.getJSONArray("activities")
    }

    fun setData(
        activities: ArrayList<LabelEntry>,
        totalStartTime: Long,
        endTime: Long,
        person: String,
        sensorMacMap: Map<String, String>
    ) {
        for (activity in activities) {
            addActivity(activity)
        }
        setTimeFinished(endTime)
        setTimeStarted(totalStartTime)
        setPerson(person)
        setSensorMacMap(sensorMacMap)
        save()
    }

    fun setVideoCaptureStartedTime(timeStarted: Long, save: Boolean = false) {
        json.put("videoCaptureStartTime", timeStarted)
        if (save) {
            save()
        }
    }

    fun setPoseCaptureStartedTime(timeStarted: Long, save: Boolean = false) {
        json.put("poseCaptureStartTime", timeStarted)
        if (save) {
            save()
        }
    }

    fun setActivities(activities: ArrayList<LabelEntry>, save: Boolean = false) {
        clearActivities()
        for (activity in activities) {
            addActivity(activity)
        }
        if (save) {
            save()
        }
    }

    fun getActivities(): ArrayList<LabelEntry> {
        val result = ArrayList<LabelEntry>()
        for (i in 0 until activities.length()) {
            val activityObj = activities[i] as JSONObject
            val timeStarted = activityObj.getLong("timeStarted")
            val label = activityObj.getString("label")
            result.add(LabelEntry(timeStarted, label))
        }
        return result
    }

    fun getVideoCaptureStartedTime(): Long? {
        val v = json.optLong("videoCaptureStartTime")
        return if (v != 0L) v else null
    }

    fun getPoseCaptureStartedTime(): Long? {
        val v = json.optLong("poseCaptureStartTime")
        return if (v != 0L) v else null
    }

    fun getDuration(): Long {
        return getTimeEnded() - getTimeStarted()
    }

    fun getTimeStarted(): Long {
        return json.getLong("startTimestamp")
    }

    fun getTimeEnded(): Long {
        return json.getLong("endTimestamp")
    }

    fun getPerson(): String {
        return json.getString("person")
    }

    fun setRecordingState(state: String) {
        json.put("recordingState", state)
        save()
    }

    fun getRecordingState(): String? {
        return try {
            json.getString("recordingState")
        } catch (exception: JSONException) {
            null
        }
    }

    fun hasBeenUsedForOnDeviceTraining(): Boolean {
        if (json.has(ON_DEVICE_TRAINING_METADATA_KEY)) {
            return json.get(ON_DEVICE_TRAINING_METADATA_KEY) as Boolean
        }
        return false
    }

    fun setUsedForOnDeviceTraining(save: Boolean = true) {
        if (!json.has(ON_DEVICE_TRAINING_METADATA_KEY)) {
            json.put(ON_DEVICE_TRAINING_METADATA_KEY, JSONObject())
        }
        val onDeviceTrainingObj = json.getJSONObject(ON_DEVICE_TRAINING_METADATA_KEY)
        val macAddress = ""
        if (!onDeviceTrainingObj.has(macAddress)) {
            val deviceInfoObj = JSONObject()
            deviceInfoObj.put("device_model", Build.MODEL)
            deviceInfoObj.put("device", Build.DEVICE)
            deviceInfoObj.put("trainingHistory", JSONArray())
            onDeviceTrainingObj.put(macAddress, deviceInfoObj)
        }
        val deviceInfoObj = onDeviceTrainingObj.getJSONObject(macAddress)
        val trainingHistory = deviceInfoObj.getJSONArray("trainingHistory")
        trainingHistory.put(System.currentTimeMillis())
        if (save) {
            save()
        }
    }

    private fun clearActivities() {
        while (activities.length() > 0) {
            activities.remove(0)
        }
    }

    private fun addActivity(activity: LabelEntry) {
        val obj = JSONObject()
        obj.put("timeStarted", activity.timeStarted)
        obj.put("label", activity.activity)
        activities.put(obj)
    }

    private fun setTimeStarted(startTime: Long) {
        json.put("startTimestamp", startTime)
    }

    private fun setTimeFinished(endTime: Long) {
        json.put("endTimestamp", endTime)
    }

    private fun setPerson(person: String) {
        json.put("person", person)
    }

    private fun setSensorMacMap(sensorMacMap: Map<String, String>) {
        val obj = JSONObject()
        for (entry in sensorMacMap.entries) {
            obj.put(entry.key, entry.value)
        }
        json.put("sensorMapping", obj)
    }
    /** Returns map of Mac address of sensors to their respective tag
     * */
    fun getSensorMacMap(): Map<String, String> {
        val result = mutableMapOf<String, String>()
        val obj = json.getJSONObject("sensorMapping")

        for (key in obj.keys()) {
            result[key] = obj.getString(key)
        }
        return result
    }

    fun clone(): RecordingMetadataStorage {
        return RecordingMetadataStorage(file, json)
    }

    fun getActivityAtTime(relativeTimeMs: Long): String? {
        if (relativeTimeMs <0) {
            return null
        }
        val activities = getActivities()
        if (relativeTimeMs> getDuration()) {
            return activities.last().activity
        }
        if (activities.isEmpty()) {
            return null
        }
        val absoluteRecStartTime = activities[0].timeStarted
        for ((index, entry) in activities.withIndex()) {
            val (absStartTime, _) = entry
            val relativeActivityStartTime = absStartTime - absoluteRecStartTime
            if (relativeTimeMs <relativeActivityStartTime) {
                return if (index> 0) (activities[index - 1].activity) else null
            }
        }
        return activities.last().activity
    }

    companion object {
        private const val ON_DEVICE_TRAINING_METADATA_KEY = "onDeviceTraining"
        fun getDurationOfActivity(
            activities: ArrayList<LabelEntry>,
            index: Int,
            recordingEndedTimeStamp: Long
        ): Long {
            // index is allowed to be equal to activities.size
            if (index >= activities.size || index < 0) {
                throw IndexOutOfBoundsException("Index $index out of bounds for activities of size ${activities.size}")
            }
            val timeStarted = activities[index].timeStarted
            val timeEnded =
                if (index + 1 < activities.size) activities[index + 1].timeStarted else recordingEndedTimeStamp
            return timeEnded - timeStarted
        }
    }
}
