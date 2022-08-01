package sensors_in_paradise.sonar.custom_views.stickman.object3d

import android.graphics.Color
import android.graphics.Paint
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec4

class Stickman : LineObject3D(
    arrayOf(
        leftFoot,
        rightFoot,
        leftKnee,
        rightKnee,
        center,
        backTop,
        throatTop,
        leftElbow,
        rightElbow,
        leftWrist,
        rightWrist
    ), arrayListOf(head)
) {

    private val linesToDraw = arrayOf(
        Pair(leftFoot, leftKnee),
        Pair(rightFoot, rightKnee),
        Pair(leftKnee, center),
        Pair(rightKnee, center),
        Pair(backTop, center),
        Pair(leftElbow, backTop),
        Pair(rightElbow, backTop),
        Pair(leftWrist, leftElbow),
        Pair(rightWrist, rightElbow),
        Pair(backTop, throatTop),
    )

    private val linePaint = Paint(0).apply {
        color = Color.BLUE
        strokeWidth = 6f
    }
    private val vectorPaint = Paint(0).apply {
        color = Color.WHITE
        strokeWidth = 9f
    }

    override fun get3DLinesToDraw(): Array<Pair<Vec4, Vec4>> {
        return linesToDraw
    }

    override fun getLinePaint(lineIndex: Int?): Paint {
        return linePaint
    }

    override fun getVectorPaint(): Paint {
        return vectorPaint
    }

    companion object {
        private val leftFoot = Vec4(-0.4f, 0f, 0f)
        private val rightFoot = Vec4(0.4f, 0f, 0f)
        private val leftKnee = Vec4(-0.3f, 0.25f, 0f)
        private val rightKnee = Vec4(0.3f, 0.25f, 0f)
        private val center = Vec4(0f, 0.5f, 0f)
        private val backTop = Vec4(0f, 1f, 0f)
        private val leftElbow = Vec4(-0.4f, +0.75f, 0f)
        private val rightElbow = Vec4(0.4f, +0.75f, 0f)
        private val leftWrist = Vec4(-0.5f, +0.5f, 0f)
        private val rightWrist = Vec4(0.5f, +0.5f, 0f)
        private val throatTop = Vec4(0f, +1.1f, 0f)
        private val head = Cube().apply {
            scale(0.15f, 0.2f, 0.15f)
            translate(0f, 1.1f, 0f)
        }
    }
}
