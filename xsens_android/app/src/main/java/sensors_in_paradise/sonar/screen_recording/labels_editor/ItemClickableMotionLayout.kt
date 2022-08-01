package sensors_in_paradise.sonar.screen_recording.labels_editor

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Matrix
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import androidx.constraintlayout.motion.widget.MotionLayout
import kotlin.math.abs

class ItemClickableMotionLayout(context: Context, attributeSet: AttributeSet) :
    MotionLayout(context, attributeSet) {
    init {
        // Set empty on click listener here so that the MotionLayout received touch events
        // even when it has only one child so it does not need to scroll
        setOnClickListener {}
    }
    private val touchSlop: Int = ViewConfiguration.get(context).scaledTouchSlop
    private var initialX = 0f
    private var initialY = 0f
    private var downTimeStamp = 0L
    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent?): Boolean {
       if (event != null) {
          if (event.actionMasked == MotionEvent.ACTION_UP) {
              if (isClick(event)) {
                       val view = findChildByPosition(event.x, event.y)
                       if (view is ClickableCarouselTextView) {
                           view.performClick()
                       }
                   }
           }
           if (event.actionMasked == MotionEvent.ACTION_DOWN) {
               initialX = event.x
               initialY = event.y
               downTimeStamp = System.currentTimeMillis()
           }
       }

        return super.onTouchEvent(event)
    }

    private fun isClick(event: MotionEvent): Boolean {
        if (abs((initialX - event.x).toInt()) <touchSlop && abs((initialY - event.y).toInt()) <touchSlop) {
            if (System.currentTimeMillis() - downTimeStamp <600L) {
                return true
            }
        }
        return false
    }

    private fun findChildByPosition(x: Float, y: Float): View? {
        val count = childCount
        for (i in count - 1 downTo 0) {
            val child = getChildAt(i)
            if (child.visibility == VISIBLE) {
                if (isPositionInChildView(child, x, y)) {
                    return child
                }
            }
        }
        return null
    }

    private val sPoint = FloatArray(2)
    private val sInvMatrix = Matrix()
    private fun isPositionInChildView(child: View, xPos: Float, yPos: Float): Boolean {
        var x = xPos
        var y = yPos
        sPoint[0] = x + scrollX - child.left
        sPoint[1] = y + scrollY - child.top
        val childMatrix: Matrix = child.matrix
        if (!childMatrix.isIdentity) {
            childMatrix.invert(sInvMatrix)
            sInvMatrix.mapPoints(sPoint)
        }
        x = sPoint[0]
        y = sPoint[1]
        return x >= 0 && y >= 0 && x < child.width && y < child.height
    }
}
