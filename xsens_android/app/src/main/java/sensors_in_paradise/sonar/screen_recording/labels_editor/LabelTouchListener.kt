package sensors_in_paradise.sonar.screen_recording.labels_editor

import android.view.MotionEvent
import android.view.View

class LabelTouchListener(val onClick: () -> Unit, val maxMsForClick: Long = 300L) : View.OnTouchListener {
    private var downTimeStamp = 0L

    override fun onTouch(v: View?, event: MotionEvent?): Boolean {
       if (event != null) {
           if (event.action == MotionEvent.ACTION_DOWN) {
               downTimeStamp = System.currentTimeMillis()
                return false
           } else if (event.action == MotionEvent.ACTION_UP) {

               if (System.currentTimeMillis() - downTimeStamp < maxMsForClick) {
                   onClick()
                   return true
               }
           }
       }
        return false
    }
}
