package sensors_in_paradise.sonar.screen_recording.camera.pose_estimation.data

import android.graphics.PointF

enum class Pose {
    BodyPose,
    HandPose;
}

data class PoseSequence(
    val timeStamps: ArrayList<Long>,
    val posesArray: ArrayList<List<List<PointF>?>>,
    val startTime: Long,
    val type: Pose
)
