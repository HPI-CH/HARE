package sensors_in_paradise.sonar.custom_views.stickman.object3d

import android.graphics.Color
import android.graphics.Paint
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec4

open class Cube(children: ArrayList<Object3D> = ArrayList()) : LineObject3D(
    arrayOf(
        lbb.clone(),
        lbf.clone(),
        ltb.clone(),
        ltf.clone(),
        rbb.clone(),
        rbf.clone(),
        rtb.clone(),
        rtf.clone()
    ), children
) {
    private val _lbb = vertices[0]
    private val _lbf = vertices[1]
    private val _ltb = vertices[2]
    private val _ltf = vertices[3]
    private val _rbb = vertices[4]
    private val _rbf = vertices[5]
    private val _rtb = vertices[6]
    private val _rtf = vertices[7]
    private val linePaint = Paint(0).apply {
        color = Color.GRAY
        strokeWidth = 3f
    }
    private val vectorPaint = Paint(0).apply {
        color = Color.WHITE
        strokeWidth = 5f
    }
    private val linesToDraw = arrayOf(
        Pair(_lbb, _lbf), Pair(_lbf, _rbf), Pair(_rbf, _rbb), Pair(_rbb, _lbb),
        Pair(_ltb, _ltf), Pair(_ltf, _rtf), Pair(_rtf, _rtb), Pair(_rtb, _ltb),
        Pair(_lbb, _ltb), Pair(_lbf, _ltf), Pair(_rbf, _rtf), Pair(_rbb, _rtb)
    )

    override fun get3DLinesToDraw(): Array<Pair<Vec4, Vec4>> {
        return linesToDraw
    }

    override fun getLinePaint(lineIndex: Int?): Paint {
        return linePaint
    }

    override fun getVectorPaint(): Paint {
        return vectorPaint
    }

    fun getTopSide(): Array<Vec4> {
        return arrayOf(_ltb, _ltf, _rtb, _rtf)
    }

	fun getNegativeXSide(): Array<Vec4> {
        return arrayOf(_lbb, _lbf, _ltb, _ltf)
    }

    companion object {
        // lbb for left-bottom-back
        private val lbb = Vec4(-0.5f, 0f, 0.5f)
        private val lbf = Vec4(-0.5f, 0f, -0.5f)
        private val ltb = Vec4(-0.5f, 1f, 0.5f)
        private val ltf = Vec4(-0.5f, 1f, -0.5f)
        private val rbb = Vec4(0.5f, 0f, 0.5f)
        private val rbf = Vec4(0.5f, 0f, -0.5f)
        private val rtb = Vec4(0.5f, 1f, 0.5f)
        private val rtf = Vec4(0.5f, 1f, -0.5f)
    }
}
