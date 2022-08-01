package sensors_in_paradise.sonar.custom_views.stickman.object3d

import android.graphics.*
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec4

class CoordinateSystem3D : LineObject3D(
    arrayOf(
        origin.clone(), xAxis.clone(), yAxis.clone(), zAxis.clone()
    )
) {

    private val linesToDraw = arrayOf(
        Pair(vertices[0], vertices[1]),
        Pair(vertices[0], vertices[2]),
        Pair(vertices[0], vertices[3]),
    )
    private val boldTypeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    private val linePaints = arrayOf(Paint(0).apply {
        color = Color.RED
        strokeWidth = 6f
        textSize = 40f
        typeface = boldTypeface
    }, Paint(0).apply {
        color = Color.GREEN
        strokeWidth = 6f
        textSize = 40f
        typeface = boldTypeface
    },
        Paint(0).apply {
            color = Color.BLUE
            strokeWidth = 6f
            textSize = 40f
            typeface = boldTypeface
        })
    private val vectorPaint = Paint(0).apply {
        color = Color.GREEN
        strokeWidth = 9f
    }

    override fun get3DLinesToDraw(): Array<Pair<Vec4, Vec4>> {
        return linesToDraw
    }

    override fun getLinePaint(lineIndex: Int?): Paint {
        return linePaints[lineIndex!!]
    }

    override fun getVectorPaint(): Paint {
        return vectorPaint
    }

    override fun drawSelf(canvas: Canvas, projectPoint: (p: Vec4) -> PointF) {
        super.drawSelf(canvas, projectPoint)

        canvas.apply {
            drawText(
                "x",
                width - 60f,
                40f,
                linePaints[0]
            )
            drawText(
                "y",
                width - 40f,
                40f,
                linePaints[1]
            )
            drawText(
                "z",
                width - 20f,
                40f,
                linePaints[2]
            )
        }
    }

    companion object {
        private val origin = Vec4(0f, 0f, 0f)
        private val xAxis = Vec4(1f, 0f, 0f)
        private val yAxis = Vec4(0f, 1f, 0f)
        private val zAxis = Vec4(0f, 0f, 1f)
    }
}
