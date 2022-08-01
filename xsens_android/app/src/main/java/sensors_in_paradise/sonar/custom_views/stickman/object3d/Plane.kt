package sensors_in_paradise.sonar.custom_views.stickman.object3d

import android.graphics.Color
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec4

class Plane(corner1: Vec4, corner2: Vec4, corner3: Vec4, corner4: Vec4, color: Int = Color.LTGRAY) :
    TriangleObject3D(arrayOf(corner1, corner2, corner3, corner4), color) {
    constructor() : this(lbb.clone(), lbf.clone(), rbb.clone(), rbf.clone())

    private val trianglesToDraw =
        arrayOf(Triple(corner1, corner2, corner4), Triple(corner1, corner4, corner3))

    override fun get3DTrianglesToDraw(): Array<Triple<Vec4, Vec4, Vec4>> {
        return trianglesToDraw
    }

    override fun hasChanged(): Boolean {
        return false
    }

    override fun onDrawn() {}

    companion object {
        // lbb for left-bottom-back
        private val lbb = Vec4(-1f, 0f, 1f)
        private val lbf = Vec4(-1f, 0f, -1f)
        private val rbb = Vec4(1f, 0f, 1f)
        private val rbf = Vec4(1f, 0f, -1f)
    }
}
