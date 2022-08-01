package sensors_in_paradise.sonar.screen_recording.labels_editor

import android.os.Handler
import android.os.Looper
import sensors_in_paradise.sonar.IntervalLooper

abstract class IntervalLoopSeeker : IntervalLooper(1000L / VisualSequenceViewHolder.FPS, false) {

    private var currentIntervalStartTime = 0L
    private val uiHandler = Handler(Looper.getMainLooper())

    private var currentInterval: Pair<Long, Long>? = null
    private var timeInInterval = 0L
    abstract fun seekTo(ms: Long)

    fun loopInterval(msStart: Long, msEnd: Long, relativeStartPosition: Long = 0L) {
        currentInterval = Pair(msStart, msEnd)
        currentIntervalStartTime = System.currentTimeMillis() - relativeStartPosition
        startLooping {
            timeInInterval = System.currentTimeMillis() - currentIntervalStartTime
            if (timeInInterval + msStart > msEnd) {
                currentIntervalStartTime = System.currentTimeMillis()
            }
            uiHandler.post {
                seekTo(msStart + timeInInterval)
            }
        }
    }

    fun resumeLooping(relativeStartPosition: Long = timeInInterval): Boolean {
        if (currentInterval == null) {
            return false
        }
        loopInterval(currentInterval!!.first, currentInterval!!.second, relativeStartPosition)
        return true
    }
}
