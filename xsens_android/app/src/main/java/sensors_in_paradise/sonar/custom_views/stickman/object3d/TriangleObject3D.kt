package sensors_in_paradise.sonar.custom_views.stickman.object3d

import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.PointF
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec4

abstract class TriangleObject3D(vertices: Array<Vec4>, val fillColor: Int) : Object3D(vertices) {
    abstract fun get3DTrianglesToDraw(): Array<Triple<Vec4, Vec4, Vec4>>
    abstract fun hasChanged(): Boolean
    abstract fun onDrawn()

    private val emptyPaint = Paint(0)
    private var colors: IntArray? = null
    private fun getTriangleVertexArray(projectPoint: (p: Vec4) -> PointF): FloatArray {
        val triangles = get3DTrianglesToDraw()
        val result = FloatArray(triangles.size * 3 * 2)

        for ((t, triangle) in triangles.withIndex()) {
            val index = t * 3 * 2
            val p1 = projectPoint(triangle.first)
            val p2 = projectPoint(triangle.second)
            val p3 = projectPoint(triangle.third)
            result[index] = p1.x
            result[index + 1] = p1.y
            result[index + 2] = p2.x
            result[index + 3] = p2.y
            result[index + 4] = p3.x
            result[index + 5] = p3.y
        }
        return result
    }

    override fun drawSelf(canvas: Canvas, projectPoint: (p: Vec4) -> PointF) {

        val triangleVertexArray = getTriangleVertexArray(projectPoint)

        if (colors == null || hasChanged()) {
            colors = IntArray(triangleVertexArray.size / 2) { _ -> fillColor }
        }

        canvas.apply {
            drawVertices(
                Canvas.VertexMode.TRIANGLES,
                triangleVertexArray.size,
                triangleVertexArray,
                0,
                null,
                0,
                colors,
                0,
                null,
                0,
                0,
                emptyPaint
            )
        }
        onDrawn()
    }
}
