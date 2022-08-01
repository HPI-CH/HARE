package sensors_in_paradise.sonar.screen_recording.camera.pose_estimation

import android.content.Context
import android.graphics.*
import android.media.Image
import android.view.TextureView
import sensors_in_paradise.sonar.R
import sensors_in_paradise.sonar.screen_recording.LoggingManager
import sensors_in_paradise.sonar.screen_recording.camera.pose_estimation.data.Person
import sensors_in_paradise.sonar.screen_recording.camera.pose_estimation.data.Pose
import java.time.LocalDateTime

class ImageProcessor(
    private val context: Context,
    private val poseEstimationStorageManager: PoseEstimationStorageManager,
    val bodyPoseDetector: PoseDetector? = null,
    val handPoseDetector: HandDetector? = null
) {
    companion object {
        /** Threshold for confidence score. */
        private const val MIN_CONFIDENCE = .4f
        const val POSE_ESTIMATION_FREQUENCY = 25

        const val INPUT_WIDTH = 1280
        const val INPUT_HEIGHT = 720
    }

    private val lock = Any()

    private fun imageToBitmap(image: Image): Bitmap {
        val buffer = image.planes[0].buffer

        val pixelStride: Int = image.planes[0].pixelStride
        val rowStride: Int = image.planes[0].rowStride
        val rowPadding = rowStride - pixelStride * image.width

        val bitmap = Bitmap.createBitmap(
            image.width + rowPadding / pixelStride,
            image.height,
            Bitmap.Config.ARGB_8888
        )
        bitmap.copyPixelsFromBuffer(buffer)
        return bitmap
    }

    private fun extractPoses(bitmap: Bitmap): List<Person> {
        val persons = mutableListOf<Person>()
        synchronized(lock) {
            bodyPoseDetector!!.estimatePoses(bitmap).let {
                persons.addAll(it)
            }
        }
        return persons.filter { it.score > MIN_CONFIDENCE }.toList()
    }

    fun clearView(overlayView: TextureView) {
        val surfaceCanvas = overlayView.lockCanvas()
        surfaceCanvas?.let { canvas ->
            VisualizationUtils.drawSkeleton(listOf(), canvas)
            overlayView.unlockCanvasAndPost(canvas)
        }
    }

    private fun drawOnCanvas(
        points: List<List<PointF>>,
        lines: List<Pair<Int, Int>>,
        overlayView: TextureView,
        bitmap: Bitmap,
        isRotated90: Boolean
    ) {
        val surfaceCanvas = overlayView.lockCanvas()
        surfaceCanvas?.let { canvas ->
            VisualizationUtils.transformPoints(
                points, bitmap, canvas,
                VisualizationUtils.Transformation.PROJECT_ON_CANVAS,
                isRotated90
            )
            VisualizationUtils.drawSkeleton(
                points,
                canvas,
                lines,
                circleColor = context.getColor(R.color.stickmanJoints),
                lineColor = context.getColor(R.color.slightBackgroundContrast)
            )
            overlayView.unlockCanvasAndPost(canvas)
        }
    }

    private fun processBodyPoseImage(
        bitmap: Bitmap,
        overlayView: TextureView,
        isRotated90: Boolean
    ) {
        val timeStamp = LoggingManager.normalizeTimeStamp(LocalDateTime.now())

        val persons = extractPoses(bitmap)

        VisualizationUtils.transformKeyPoints(
            persons, bitmap,
            VisualizationUtils.Transformation.NORMALIZE
        )
        if (isRotated90) {
            VisualizationUtils.transformKeyPoints(
                persons, null,
                VisualizationUtils.Transformation.ROTATE90
            )
        }
        poseEstimationStorageManager.storeBodyPoses(persons, timeStamp)

        val pointLists = VisualizationUtils.convertTo2DPoints(persons)
        val lines = VisualizationUtils.get2DLines(Pose.BodyPose)
        drawOnCanvas(pointLists, lines, overlayView, bitmap, isRotated90)
    }

    private fun processHandPoseImage(
        bitmap: Bitmap,
        overlayView: TextureView,
        isRotated90: Boolean
    ) {
        val timeStamp = LoggingManager.normalizeTimeStamp(LocalDateTime.now())

        handPoseDetector!!.estimatePose(bitmap) { handsResult ->
            var hands = handsResult.multiHandLandmarks().toList()
            val handsClasses = handsResult.multiHandedness().map { handClass ->
                // Because of camera mirroring, the sides are inverted
                when (handClass.label) {
                    "Right" -> "Left"
                    "Left" -> "Right"
                    else -> { handClass.label }
                }
            }
            if (isRotated90) {
                hands = VisualizationUtils.transformHandLandmarks(
                    hands, null,
                    VisualizationUtils.Transformation.ROTATE90
                )
            }
            poseEstimationStorageManager.storeHandPoses(hands, handsClasses, timeStamp)

            val pointLists = VisualizationUtils.convertTo2DPoints(hands)
            val lines = VisualizationUtils.get2DLines(Pose.HandPose)
            drawOnCanvas(pointLists, lines, overlayView, bitmap, isRotated90)
        }
    }

    fun processImage(bitmap: Bitmap, overlayView: TextureView, isRotated90: Boolean) {
        if (bodyPoseDetector != null) processBodyPoseImage(bitmap, overlayView, isRotated90)
        if (handPoseDetector != null) processHandPoseImage(bitmap, overlayView, isRotated90)
    }

    fun processImage(image: Image, overlayView: TextureView, isRotated90: Boolean) {
        processImage(imageToBitmap(image), overlayView, isRotated90)
    }
}
