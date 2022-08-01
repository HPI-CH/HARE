package sensors_in_paradise.sonar.custom_views.stickman

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.os.Looper
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import androidx.annotation.AnyThread
import sensors_in_paradise.sonar.custom_views.stickman.math.Matrix4x4
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec3
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec4
import sensors_in_paradise.sonar.custom_views.stickman.object3d.Object3D
import sensors_in_paradise.sonar.custom_views.stickman.object3d.OnObjectChangedInterface

class Render3DView(
    context: Context,
    attrs: AttributeSet?,
    private val drawDebugInfo: Boolean = false
) : View(context, attrs), OnObjectChangedInterface {
    constructor(context: Context, drawDebugInfo: Boolean = false) : this(
        context,
        null,
        drawDebugInfo
    )

    constructor(context: Context, attrs: AttributeSet?) : this(context, attrs, false)

    val objects3DToDraw = ArrayList<Object3D>()
    var enableYRotation = false
    var showFPS = false
    val camera = Camera(Vec3(0f, 0f, 0f), Vec3(0f, 0f, 2f), Vec3(0f, 1f, 0f), this::onSceneChanged)
    private val projection = Matrix4x4.project(90f, 1f, 0.1f, 5f)
    private val fpsTextPaint = Paint(0).apply {
        color = Color.WHITE
        textSize = 30F
    }
    private var lastTimeSceneChanged = 0L
    private var centerX = 1f
    private var centerY = 1f
    private val minTimeMsBetweenFrames = 1000L / 60L
    private var lastTimeDrawn = 0L
    private var numDroppedFrames = 0L
    private val textBounds = Rect()
    private var drawTime = 0L
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val startTime = System.currentTimeMillis()
        canvas.apply {
            for (obj in objects3DToDraw) {
                obj.draw(canvas, this@Render3DView::project3DPoint)
            }
        }
        if (showFPS) {
            drawFPS(canvas, startTime)
        }
        val endTime = System.currentTimeMillis()
        drawTime = endTime - startTime
        lastTimeDrawn = System.currentTimeMillis()
    }

    private fun drawFPS(canvas: Canvas, drawStartTime: Long) {
        val endTime = System.currentTimeMillis()
        val fps = 1000 / (System.currentTimeMillis() - lastTimeDrawn)
        val fpsText = "FPS: $fps"

        fpsTextPaint.getTextBounds(fpsText, 0, fpsText.length, textBounds)
        canvas.apply {
            drawText(fpsText, 10f, textBounds.height().toFloat(), fpsTextPaint)
        }
        if (drawDebugInfo) {
            val droppedFramesText = "Dropped: $numDroppedFrames"
            val drawTimeText = "DrawTime: ${endTime - drawStartTime}ms"
            canvas.apply {
                drawText(droppedFramesText, 10f, textBounds.height().toFloat() + 30f, fpsTextPaint)
                drawText(drawTimeText, 10f, textBounds.height().toFloat() + 60f, fpsTextPaint)
            }
        }
    }

    private fun project3DPoint(vec4: Vec4): PointF {
        val projected = projection * camera.lookAtMatrix * vec4
        val p = (projected.xy / projected.w + 1f) * 0.5f

        return PointF(p.x * width.toFloat(), height - p.y * height.toFloat())
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        centerX = (w / 2).toFloat()
        centerY = (h / 2).toFloat()
        Matrix4x4.project(projection, 90f, w.toFloat() / h.toFloat(), 0.1f, 4f)
    }

    private var lastEventX = 0f

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent?): Boolean {
        if (enableYRotation) {
            if (event != null) {
                if (event.action == MotionEvent.ACTION_MOVE) {
                    val x = event.getAxisValue(MotionEvent.AXIS_X)
                    val diff = lastEventX - x
                    lastEventX = x
                    camera.rotateY(diff / 5f)
                    onSceneChanged()
                }
                if (event.action == MotionEvent.ACTION_DOWN) {
                    lastEventX = event.getAxisValue(MotionEvent.AXIS_X)
                }
            }
        }
        return enableYRotation
    }

    @AnyThread
    private fun onSceneChanged() {
        lastTimeSceneChanged = System.currentTimeMillis()
        if (System.currentTimeMillis() - lastTimeDrawn > minTimeMsBetweenFrames - drawTime) {
            Looper.getMainLooper().run { invalidate() }
        } else {
            numDroppedFrames++
        }
    }

    fun addObject3D(obj: Object3D) {
        objects3DToDraw.add(obj)
        obj.onObjectChanged = this
        onSceneChanged()
    }

    override fun onObjectChanged() {
        onSceneChanged()
    }
}
