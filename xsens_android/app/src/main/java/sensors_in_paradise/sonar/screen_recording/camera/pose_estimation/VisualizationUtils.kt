/*
Code from: https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/android
==============================================================================
Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

package sensors_in_paradise.sonar.screen_recording.camera.pose_estimation

import android.graphics.*
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList
import com.google.mediapipe.solutions.hands.Hands
import sensors_in_paradise.sonar.screen_recording.camera.pose_estimation.data.*

object VisualizationUtils {
    /** Radius of circle used to draw keypoints.  */
    private const val CIRCLE_RADIUS = 8f

    /** Width of line used to connected two keypoints.  */
    private const val LINE_WIDTH = 6f

    enum class Transformation {
        NORMALIZE,
        ROTATE90,
        PROJECT_ON_CANVAS,
    }

    private fun normalizePoint(p: PointF, inputSize: PointF): PointF {
        return PointF(p.x / inputSize.x, p.y / inputSize.y)
    }

    private fun rotatePoint90(p: PointF): PointF {
        return PointF(1f - p.y, p.x)
    }

    private fun projectPointOnCanvas(
        p: PointF,
        inputSize: PointF,
        outputSize: PointF,
        isRotated: Boolean
    ): PointF {
        // p should be normalized to [0,1]
        val actualInputSize = if (isRotated) PointF(inputSize.y, inputSize.x) else inputSize
        val heightRatio = (actualInputSize.y * outputSize.x) / (outputSize.y * actualInputSize.x)
        val pScaled = PointF(p.x, (p.y * heightRatio) - ((heightRatio - 1f) / 2f))
        return PointF(pScaled.x * outputSize.x, pScaled.y * outputSize.y)
    }

    @JvmName("convertTo2DPoints1")
    fun convertTo2DPoints(
        persons: List<Person>
    ): List<List<PointF>> {
        return persons.map { person ->
            person.keyPoints.map { keyPoint ->
                keyPoint.coordinate
            }
        }
    }

    @JvmName("convertTo2DPoints2")
    fun convertTo2DPoints(
        hands: List<NormalizedLandmarkList>
    ): List<List<PointF>> {
        return hands.map { hand ->
            hand.landmarkList.map { landmark ->
                PointF(landmark.x, landmark.y)
            }
        }
    }

    fun get2DLines(poseType: Pose): List<Pair<Int, Int>> {
        return when (poseType) {
            Pose.BodyPose -> Person.BODY_JOINTS.map { bodyParts ->
                Pair(bodyParts.first.position, bodyParts.second.position)
            }

            Pose.HandPose -> Hands.HAND_CONNECTIONS.map { handConnection ->
                Pair(handConnection.start(), handConnection.end())
            }
        }
    }

    fun transformPoints(
        pointLists: List<List<PointF>>,
        bitmap: Bitmap?,
        canvas: Canvas?,
        transformation: Transformation,
        isRotated: Boolean = false
    ) {
        val inputSize = bitmap?.let { PointF(bitmap.width.toFloat(), bitmap.height.toFloat()) }
        val outputSize = canvas?.let { PointF(canvas.width.toFloat(), canvas.height.toFloat()) }
        pointLists.forEach { points ->
            points.forEach { point ->
                when (transformation) {
                    Transformation.NORMALIZE -> {
                        val newPoint = normalizePoint(point, inputSize!!)
                        point.x = newPoint.x; point.y = newPoint.y
                    }

                    Transformation.ROTATE90 -> {
                        val newPoint = rotatePoint90(point)
                        point.x = newPoint.x; point.y = newPoint.y
                    }

                    Transformation.PROJECT_ON_CANVAS -> {
                        val newPoint = projectPointOnCanvas(
                            point,
                            inputSize!!,
                            outputSize!!,
                            isRotated
                        )
                        point.x = newPoint.x; point.y = newPoint.y
                    }
                }
            }
        }
    }

    fun transformKeyPoints(
        persons: List<Person>,
        bitmap: Bitmap?,
        transformation: Transformation
    ) {
        val inputSize = bitmap?.let { PointF(bitmap.width.toFloat(), bitmap.height.toFloat()) }
        persons.forEach { person ->
            person.keyPoints.forEach { keyPoint ->
                when (transformation) {
                    Transformation.NORMALIZE -> keyPoint.coordinate =
                        normalizePoint(keyPoint.coordinate, inputSize!!)

                    Transformation.ROTATE90 -> keyPoint.coordinate =
                        rotatePoint90(keyPoint.coordinate)
                    else -> {}
                }
            }
        }
    }

    fun transformHandLandmarks(
        hands: List<NormalizedLandmarkList>,
        bitmap: Bitmap?,
        transformation: Transformation
    ): List<NormalizedLandmarkList> {
        val inputSize = bitmap?.let { PointF(bitmap.width.toFloat(), bitmap.height.toFloat()) }
        return hands.map { hand ->
            val listBuilder = hand.toBuilder()
            val landmarkList = hand.landmarkList.map { landMark ->
                val landMarkBuilder = landMark.toBuilder()
                val oldCoordinate = PointF(landMark.x, landMark.y)
                val newCoordinate = when (transformation) {
                    Transformation.NORMALIZE -> normalizePoint(oldCoordinate, inputSize!!)
                    Transformation.ROTATE90 -> rotatePoint90(oldCoordinate)
                    else -> {
                        oldCoordinate
                    }
                }
                landMarkBuilder.x = newCoordinate.x; landMarkBuilder.y = newCoordinate.y
                landMarkBuilder.build()
            }
            listBuilder
                .clearLandmark()
                .addAllLandmark(landmarkList)
                .build()
        }
    }

    @Suppress("ComplexCondition")
    fun interpolatePose(
        posesArray: ArrayList<List<PointF>>,
        timeStamps: ArrayList<Long>,
        floorIndex: Int,
        timeStamp: Long,
        timeMargin: Long = 800
    ): List<PointF>? {
        val priorTimestamp = timeStamps.getOrNull(floorIndex) ?: Long.MAX_VALUE
        val nextTimestamp = timeStamps.getOrNull(floorIndex + 1)

        val priorPose = posesArray.getOrNull(floorIndex)
        val nextPose = posesArray.getOrNull(floorIndex + 1)

        if (priorPose == null ||
            timeStamp < priorTimestamp ||
            timeStamp > nextTimestamp ?: Long.MAX_VALUE ||
            (timeStamp - priorTimestamp) > timeMargin
        ) { // No prior sample / TimeStamp inconsistent / big gap to prior AND next sample
            return null
        } else if (nextPose == null ||
            nextTimestamp == null ||
            (nextTimestamp - priorTimestamp) > timeMargin
        ) { // No next sample / big gap to next sample
            return priorPose
        } else {
            val interpolateFactor =
                (timeStamp - priorTimestamp).toFloat() / (nextTimestamp - priorTimestamp).toFloat()
            val posePoints = priorPose.indices.map { pIndex ->
                val priorPoint = priorPose[pIndex]
                val nextPoint = nextPose[pIndex]

                val x = priorPoint.x + (nextPoint.x - priorPoint.x) * interpolateFactor
                val y = priorPoint.y + (nextPoint.y - priorPoint.y) * interpolateFactor

                PointF(x, y)
            }.toList()
            return posePoints
        }
    }

    // Draw line and point indicating pose
    fun drawSkeleton(
        pointLists: List<List<PointF>>,
        canvas: Canvas,
        lines: List<Pair<Int, Int>> = listOf(),
        clearColor: Int? = null,
        circleColor: Int = Color.BLACK,
        circleColors: List<Int?>? = null,
        lineColor: Int = Color.WHITE,
        circleRadius: Float = CIRCLE_RADIUS,
        lineWidth: Float = LINE_WIDTH
    ) {
        val paintCircle = { index: Int ->
            Paint().apply {
                strokeWidth = circleRadius
                color = circleColors?.getOrElse(index) { lineColor } ?: circleColor
                style = Paint.Style.FILL_AND_STROKE
            }
        }
        val paintLine = Paint().apply {
            strokeWidth = lineWidth
            color = lineColor
            style = Paint.Style.STROKE
        }

        if (clearColor == null) {
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        } else {
            canvas.drawColor(clearColor)
        }

        pointLists.forEach { points ->
            lines.forEach { line ->
                val pointA = points[line.first]
                val pointB = points[line.second]
                canvas.drawLine(pointA.x, pointA.y, pointB.x, pointB.y, paintLine)
            }

            points.forEachIndexed { index, point ->
                canvas.drawCircle(
                    point.x,
                    point.y,
                    circleRadius,
                    paintCircle(index)
                )
            }
        }
    }
}
