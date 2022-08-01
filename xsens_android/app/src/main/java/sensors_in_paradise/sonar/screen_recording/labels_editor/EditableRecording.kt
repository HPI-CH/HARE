package sensors_in_paradise.sonar.screen_recording.labels_editor

import sensors_in_paradise.sonar.GlobalValues
import sensors_in_paradise.sonar.screen_recording.Recording
import sensors_in_paradise.sonar.screen_recording.RecordingMetadataStorage

class EditableRecording(
    private val recording: Recording,
    val onActivityInserted: (index: Int) -> Unit,
    val onActivityRemoved: (index: Int) -> Unit
) {
    private val metadataCopy = recording.metadataStorage.clone()
    val activities = metadataCopy.getActivities()
    fun getDurationOfActivity(index: Int): Long {
        val startTime = getRelativeStartTimeOfActivity(index)
        val endTime = getRelativeEndTimeOfActivity(index)
        return endTime - startTime
    }

    private fun getActualTimeStarted(): Long {
        return activities[0].timeStarted
    }

	fun save() {
        metadataCopy.setActivities(activities, true)
        recording.metadataStorage = metadataCopy
    }

    fun getDuration(): Long {
        return metadataCopy.getTimeEnded() - getActualTimeStarted()
    }

    fun getRelativeStartTimeOfActivity(index: Int): Long {
        return if (index < 0) 0L else activities[index].timeStarted - getActualTimeStarted()
    }

    fun getRelativeEndTimeOfActivity(index: Int): Long {
        return if (index + 1 < activities.size) {
            activities[index + 1].timeStarted - getActualTimeStarted()
        } else {
            metadataCopy.getTimeEnded() - getActualTimeStarted()
        }
    }

    private fun insertActivity(index: Int, relativeStartTime: Long, label: String) {
        activities.add(
            index,
            RecordingMetadataStorage.LabelEntry(relativeStartTime + getActualTimeStarted(), label)
        )
        onActivityInserted(index)
    }

    fun splitActivity(index: Int, recordingRelativeSplitTime: Long) {
        val activityToSplit = activities[index]
        insertActivity(
            index + 1,
            recordingRelativeSplitTime,
            activityToSplit.activity
        )
    }

    fun setRelativeStartTimeOfActivity(index: Int, relativeStartTime: Long) {
        val timeRecordingStarted = getActualTimeStarted()
        activities[index].timeStarted = timeRecordingStarted + relativeStartTime
        if (index == 0) {
            activities.add(
                0,
                RecordingMetadataStorage.LabelEntry(
                    timeRecordingStarted,
                    GlobalValues.NULL_ACTIVITY
                )
            )
            onActivityInserted(0)
        } else if (index > 0) {
            val startTimeOfPreviousActivity = getRelativeStartTimeOfActivity(index - 1)
            if (startTimeOfPreviousActivity == relativeStartTime) {
                activities.removeAt(index - 1)
                onActivityRemoved(index - 1)
            }
        }
    }

    fun setRelativeEndTimeOfActivity(index: Int, relativeEndTime: Long) {
        val timeRecordingStarted = getActualTimeStarted()
        if (index + 1 < activities.size) {
            val timeNextRecordingEnds = getRelativeEndTimeOfActivity(index + 1)
            if (timeNextRecordingEnds == relativeEndTime) {
                activities.removeAt(index + 1)
                onActivityRemoved(index + 1)
            } else {
                activities[index + 1].timeStarted = timeRecordingStarted + relativeEndTime
            }
        } else {
            activities.add(
                RecordingMetadataStorage.LabelEntry(
                    timeRecordingStarted + relativeEndTime,
                    GlobalValues.NULL_ACTIVITY
                )
            )
            onActivityInserted(activities.size - 1)
        }
    }

    fun relativeSensorTimeToVideoTime(relativeSensorTime: Long): Long {
        val videoStartTime = metadataCopy.getVideoCaptureStartedTime()
        if (videoStartTime != null) {
            val videoDelay = videoStartTime - getActualTimeStarted()
            return relativeSensorTime - videoDelay
        }
        return relativeSensorTime
    }

    fun relativeSensorTimeToPoseSequenceTime(relativeSensorTime: Long): Long {
        val videoStartTime = metadataCopy.getVideoCaptureStartedTime()
        if (videoStartTime != null) {
            val videoDelay = videoStartTime - getActualTimeStarted()
            return relativeSensorTime - videoDelay
        }
        return relativeSensorTime
    }

    fun areTimeStampsConsistent(): Boolean {
        var lastTime = 0L
        for (activity in activities) {
            if (lastTime > activity.timeStarted) {
                return false
            }
            lastTime = activity.timeStarted
        }
        return true
    }
}
