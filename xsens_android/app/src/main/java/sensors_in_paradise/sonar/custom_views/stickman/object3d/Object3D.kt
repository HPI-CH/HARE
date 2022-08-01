package sensors_in_paradise.sonar.custom_views.stickman.object3d

import android.graphics.*
import sensors_in_paradise.sonar.custom_views.stickman.math.Matrix4x4
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec4
import java.util.*
import kotlin.collections.ArrayList
import kotlin.math.min

abstract class Object3D(
    protected val vertices: Array<Vec4>,
    protected val children: ArrayList<Object3D> = ArrayList(),
    var onObjectChanged: OnObjectChangedInterface? = null
) {
    private val defaultVertices = vertices.map { it.clone() }
    var drawVertexPositionsForDebugging = false
    private val vertexPositionsDebuggingHeader = "i_|_x__|_y__|_z__"
    private val debugTextBounds = Rect().apply {
        debugTextPaint.getTextBounds(
            vertexPositionsDebuggingHeader,
            0,
            vertexPositionsDebuggingHeader.length,
            this
        )
    }

    fun scale(x: Float, y: Float, z: Float, shouldNotifyThatVerticesChanged: Boolean = true) {
        val m = Matrix4x4().apply { scale(x, y, z) }
        applyOnAllVertices(m, shouldNotifyThatVerticesChanged = shouldNotifyThatVerticesChanged)
    }

    fun translate(x: Float, y: Float, z: Float, shouldNotifyThatVerticesChanged: Boolean = true) {
        val m = Matrix4x4().apply { translate(x, y, z) }
        applyOnAllVertices(m, shouldNotifyThatVerticesChanged = shouldNotifyThatVerticesChanged)
    }

    fun rotate(
        degrees: Float,
        xFactor: Float,
        yFactor: Float,
        zFactor: Float,
        shouldNotifyThatVerticesChanged: Boolean = true
    ) {
        val m = Matrix4x4().apply { this.rotate(degrees, xFactor, yFactor, zFactor) }
        applyOnAllVertices(m, shouldNotifyThatVerticesChanged = shouldNotifyThatVerticesChanged)
    }

    fun rotateEuler(
        xDegrees: Float,
        yDegrees: Float,
        zDegrees: Float,
        shouldNotifyThatVerticesChanged: Boolean = true
    ) {
        val m = Matrix4x4.rotateEuler(xDegrees, yDegrees, zDegrees)
        applyOnAllVertices(m, shouldNotifyThatVerticesChanged = shouldNotifyThatVerticesChanged)
    }

    fun rotateEulerRadians(
        x: Float,
        y: Float,
        z: Float,
        shouldNotifyThatVerticesChanged: Boolean = true
    ) {
        rotateEuler(
            radiansToDegrees(x),
            radiansToDegrees(y),
            radiansToDegrees(z), shouldNotifyThatVerticesChanged
        )
    }

    private fun radiansToDegrees(radians: Float): Float {
        return (radiansToDegreesFactor * radians)
    }

    private fun applyOnAllVertices(
        m: Matrix4x4,
        applyOnChildren: Boolean = true,
        shouldNotifyThatVerticesChanged: Boolean = true
    ) {
        for (v in vertices) {
            v *= m
        }
        if (applyOnChildren) {
            for (child in children) {
                child.applyOnAllVertices(m, false)
            }
        }
        if (shouldNotifyThatVerticesChanged) {
            notifyVerticesChanged()
        }
    }

    private fun drawVertexPositionsForDebugging(canvas: Canvas) {
        var y = 70f
        canvas.drawText(
            vertexPositionsDebuggingHeader,
            canvas.width - 10f - debugTextBounds.width(),
            y,
            debugTextPaint
        )
        val f = { x: Float -> String.format(Locale.US, "%.2f", x) }
        val i = { i: Int -> i.toString().padStart(2, ' ') }

        for (index in 0 until min(vertices.size, 99)) {
            val v = vertices[index]
            y += debugTextBounds.height()
            canvas.drawText(
                "${i(index)}|${f(v.x)}|${f(v.y)}|${f(v.z)}",
                canvas.width - 10f - debugTextBounds.width(),
                y,
                debugTextPaint
            )
        }
    }

    fun draw(canvas: Canvas, projectPoint: (p: Vec4) -> PointF) {
        if (drawVertexPositionsForDebugging) {
            drawVertexPositionsForDebugging(canvas)
        }
        drawSelf(canvas, projectPoint)

        for (child in children) {
            child.draw(canvas, projectPoint)
        }
    }

    protected abstract fun drawSelf(canvas: Canvas, projectPoint: (p: Vec4) -> PointF)

    fun notifyVerticesChanged() {
        onObjectChanged?.onObjectChanged()
    }

    /**Updates the default state of this object 3d using it's current state*/
    fun updateDefaultState(applyOnChildren: Boolean = true) {
        for (i in vertices.indices) {
            defaultVertices[i].assign(vertices[i])
        }
        if (applyOnChildren) {
            for (child in children) {
                child.updateDefaultState(
                    applyOnChildren = true
                )
            }
        }
    }

	fun resetToDefaultState(
	    applyOnChildren: Boolean = true,
	    shouldNotifyThatVerticesChanged: Boolean = true
	) {
        for (i in vertices.indices) {
            vertices[i].assign(defaultVertices[i])
        }
        if (applyOnChildren) {
            for (child in children) {
                child.resetToDefaultState(
                    applyOnChildren = true,
                    shouldNotifyThatVerticesChanged = false
                )
            }
        }
        if (shouldNotifyThatVerticesChanged) {
            notifyVerticesChanged()
        }
    }

    companion object {
        private const val radiansToDegreesFactor = Math.PI.toFloat() / 180f
        private val debugTextPaint = Paint(0).apply {
            color = Color.WHITE
        }
    }
}
