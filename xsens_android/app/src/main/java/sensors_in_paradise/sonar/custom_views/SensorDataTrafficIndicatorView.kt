package sensors_in_paradise.sonar.custom_views

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import androidx.core.view.postDelayed
import sensors_in_paradise.sonar.R
import java.lang.Integer.max

class SensorDataTrafficIndicatorView(context: Context, attrs: AttributeSet) : View(context, attrs) {

    private class TextPaint(text: String, color: Int) : Paint(0) {
        val textBounds = Rect()

        init {
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            getTextBounds(text, 0, text.length, textBounds)
            this.color = color
        }
    }

    private class TrafficIndicator(
        val rect: Rect,
        val paint: Paint,
        val text: String? = null,
        val textColor: Int = Color.BLACK
    ) {
        var lastDataReceived = 0L
        val textPaint = text?.let { TextPaint(it, textColor) }
        val textPos = textPaint?.let {
            val textBounds = it.textBounds
            val centerX = rect.exactCenterX()
            val centerY = rect.exactCenterY()
            return@let PointF(centerX - textBounds.width() / 2, centerY + textBounds.height() / 2)
        }
    }

    var numSensors = 0
        set(value) {
            field = value
            initTrafficIndicators()
            invalidate()
        }
    var pixelsBetween = 12
        set(value) {
            field = value
            invalidate()
        }
    private var indicatorColor = Color.RED
        set(value) {
            field = value
            for (indicator in trafficIndicators) {
                indicator.paint.color = value
            }
            invalidate()
        }
    private var idleColor = Color.GRAY
        set(value) {
            field = value
            idlePaint.color = value
            invalidate()
        }
    private val idlePaint = Paint(0).apply {
        color = idleColor
        alpha = 255
    }

    private val trafficIndicators: ArrayList<TrafficIndicator> = ArrayList()
    private var animationRunning = false
    private var fadeOutDuration = 1000
        set(value) {
            field = value
            animationFrameDelay = max(30, value / 255)
            invalidate()
        }
    private var animationFrameDelay = 30
    private var textColor = Color.BLACK
        set(value) {
            for (indicator in trafficIndicators) {
                indicator.textPaint?.color = value
            }
            field = value
            invalidate()
        }
    private var sensorTags: Array<String>? = null

    init {
        context.theme.obtainStyledAttributes(
            attrs,
            R.styleable.SensorDataTrafficIndicator,
            0, 0
        ).apply {

            try {
                indicatorColor = getColor(
                    R.styleable.SensorDataTrafficIndicator_indicatorColor,
                    indicatorColor
                )
                idleColor = getColor(
                    R.styleable.SensorDataTrafficIndicator_idleColor,
                    idleColor
                )
                pixelsBetween = getInt(
                    R.styleable.SensorDataTrafficIndicator_pixelsBetween,
                    pixelsBetween
                )
                fadeOutDuration = getInt(
                    R.styleable.SensorDataTrafficIndicator_fadeOutDuration,
                    fadeOutDuration
                )
                textColor = getColor(
                    R.styleable.SensorDataTrafficIndicator_textColor,
                    textColor
                )
            } finally {
                recycle()
            }
        }
    }

    fun setNumSensors(tags: Array<String>) {
        this.sensorTags = tags
        this.numSensors = tags.size
    }

    private fun initTrafficIndicators() {
        val w = width
        val h = height
        trafficIndicators.clear()
        if (numSensors> 0) {
            val widthPerSensor = ((w - paddingLeft - paddingRight) / numSensors)
            for (i in 0 until numSensors) {
                val rect = Rect(
                    paddingLeft + i * widthPerSensor + if (i != 0) pixelsBetween / 2 else 0,
                    paddingTop,
                    paddingLeft + (i + 1) * widthPerSensor - if (i != numSensors - 1) pixelsBetween / 2 else 0,
                    h - paddingBottom
                )
                val paint = Paint(0).apply {
                    color = indicatorColor
                    alpha = 0
                }
                val text = sensorTags?.get(i)
                val indicator = TrafficIndicator(rect, paint, text = text, textColor)
                trafficIndicators.add(indicator)
            }
        }
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)

        initTrafficIndicators()
        doAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.apply {
            trafficIndicators.forEach {
                drawRect(it.rect, idlePaint)
                drawRect(it.rect, it.paint)
                if (it.text != null && it.textPaint != null && it.textPos != null) {
                    drawText(it.text, it.textPos.x, it.textPos.y, it.textPaint)
                }
            }
        }
    }

    fun setSensorDataReceived(sensorIndex: Int) {
        if (sensorIndex < trafficIndicators.size) {
            trafficIndicators[sensorIndex].lastDataReceived = System.currentTimeMillis()
        }
        if (!animationRunning) {
            doAnimation()
        }
    }

    /** Returns true if there is one fading out animation still running, false otherwise*/
    private fun updatePaints(): Boolean {
        val currTime = System.currentTimeMillis()
        var result = false
        for (indicator in trafficIndicators) {
            val lastReceived = indicator.lastDataReceived
            val paint = indicator.paint

            val timeDiff = currTime - lastReceived

            if (timeDiff <= fadeOutDuration) {
                paint.alpha = 255 - ((timeDiff * 255) / fadeOutDuration).toInt()
                result = true
            } else {
                paint.alpha = 0
            }
        }
        return result
    }

    private fun doAnimation() {
        animationRunning = updatePaints()
        invalidate()
        if (animationRunning) {
            postDelayed(50) {
                doAnimation()
            }
        }
    }
}
