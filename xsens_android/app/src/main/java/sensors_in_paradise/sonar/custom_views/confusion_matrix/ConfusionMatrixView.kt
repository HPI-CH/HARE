package sensors_in_paradise.sonar.custom_views.confusion_matrix

import android.content.Context
import android.graphics.Canvas
import android.graphics.Rect
import android.util.AttributeSet
import android.util.Log
import android.view.View

class ConfusionMatrixView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    private var confusionMatrix: DrawableConfusionMatrix? = null

        set(value) {
            field = value
            invalidate()
        }

    fun setConfusionMatrix(confusionMatrix: ConfusionMatrix) {
        this.confusionMatrix = DrawableConfusionMatrix(context, confusionMatrix)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val cm = confusionMatrix
        if (cm != null) {
            val boundingBox = Rect(0, 0, width, height)

            cm.draw(canvas, boundingBox)
        }
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        Log.v("Chart onMeasure w", MeasureSpec.toString(widthMeasureSpec))
        Log.v("Chart onMeasure h", MeasureSpec.toString(heightMeasureSpec))

        val widthSpecSize = MeasureSpec.getSize(widthMeasureSpec)
        val desiredWidth = widthSpecSize
        val desiredHeight = ((confusionMatrix?.getDesiredHeightForWidth(desiredWidth))
            ?: suggestedMinimumHeight) + paddingTop + paddingBottom

        Log.d("ConfusionMatrixView-onMeasure", "desiredWidth: $desiredWidth, desiredHeight: $desiredHeight")
        setMeasuredDimension(
            measureDimension(desiredWidth, widthMeasureSpec),
            measureDimension(desiredHeight, heightMeasureSpec)
        )
    }

    private fun measureDimension(desiredSize: Int, measureSpec: Int): Int {
        var result: Int
        val specMode = MeasureSpec.getMode(measureSpec)
        val specSize = MeasureSpec.getSize(measureSpec)
        if (specMode == MeasureSpec.EXACTLY) {
            result = specSize
        } else {
            result = desiredSize
            if (specMode == MeasureSpec.AT_MOST) {
                result = Math.min(result, specSize)
            }
        }
        if (result < desiredSize) {
            Log.e("ChartView", "The view is too small, the content might get cut")
        }
        return result
    }
}
