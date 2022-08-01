package sensors_in_paradise.sonar

import android.os.Handler
import android.os.Looper
import java.util.*

open class IntervalLooper(private val intervalMs: Long, val callRunOnUiThread: Boolean = false) {
    private var loopTimerTask: TimerTask? = null
    private val timer: Timer = Timer()
    private val uiHandler = Handler(Looper.getMainLooper())

    fun startLooping(run: Runnable) {
        loopTimerTask?.cancel()
        loopTimerTask = object : TimerTask() {
            override fun run() {
               if (callRunOnUiThread) {
                   uiHandler.post(run)
               } else {
                    run.run()
                }
            }
        }
        timer.scheduleAtFixedRate(loopTimerTask, 0L, intervalMs)
    }

    fun stopLooping() {
        loopTimerTask?.cancel()
    }
}
