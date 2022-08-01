package sensors_in_paradise.sonar.custom_views.stickman

import sensors_in_paradise.sonar.custom_views.stickman.math.Matrix4x4
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec3
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec4

class Camera(val center: Vec3, eye: Vec3, val up: Vec3, private val onCameraChanged: (() -> Unit)? = null) {
    val eye = Vec4(eye)
    val lookAtMatrix = Matrix4x4.lookAt(eye, center, up)

    fun rotateY(degrees: Float, shouldNotifyThatCameraChanged: Boolean = true) {
        eye *= Matrix4x4().apply { rotateY(degrees) }
        if (shouldNotifyThatCameraChanged) {
            notifyCameraChanged()
        }
    }

	fun notifyCameraChanged() {
        Matrix4x4.lookAt(eye.xyz, center, up, lookAtMatrix)
        onCameraChanged?.let { it() }
    }
}
