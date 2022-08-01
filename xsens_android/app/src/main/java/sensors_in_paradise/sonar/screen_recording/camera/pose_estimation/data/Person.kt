/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

package sensors_in_paradise.sonar.screen_recording.camera.pose_estimation.data

import android.graphics.PointF
import android.graphics.RectF

data class Person(
    var id: Int = -1, // default id is -1
    var keyPoints: List<KeyPoint>,
    val boundingBox: RectF? = null, // Only MoveNet MultiPose return bounding box.
    val score: Float
) {
    fun copy(): Person {
        return Person(id, keyPoints.map { it.copy() }, boundingBox, score)
    }

    companion object {
        fun getNULLPerson(): Person {
            return Person(
                -1,
                BodyPart.values().map { bp -> KeyPoint(bp, PointF(-1f, -1f), 0f) },
                RectF(),
                0f
            )
        }

        /** Pair of keypoints to draw lines between.  */
        val BODY_JOINTS = listOf(
            Pair(BodyPart.NOSE, BodyPart.LEFT_EYE),
            Pair(BodyPart.NOSE, BodyPart.RIGHT_EYE),
            Pair(BodyPart.LEFT_EYE, BodyPart.LEFT_EAR),
            Pair(BodyPart.RIGHT_EYE, BodyPart.RIGHT_EAR),
            Pair(BodyPart.NOSE, BodyPart.LEFT_SHOULDER),
            Pair(BodyPart.NOSE, BodyPart.RIGHT_SHOULDER),
            Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW),
            Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST),
            Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
            Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
            Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
            Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
            Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_HIP),
            Pair(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
            Pair(BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
            Pair(BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
            Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
            Pair(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)
        )
    }
}
