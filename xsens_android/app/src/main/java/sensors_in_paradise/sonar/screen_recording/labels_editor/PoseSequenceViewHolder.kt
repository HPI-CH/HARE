package sensors_in_paradise.sonar.screen_recording.labels_editor

import android.content.Context
import android.graphics.PointF
import android.view.TextureView
import sensors_in_paradise.sonar.R
import sensors_in_paradise.sonar.screen_recording.camera.pose_estimation.PoseEstimationStorageManager
import sensors_in_paradise.sonar.screen_recording.camera.pose_estimation.VisualizationUtils
import sensors_in_paradise.sonar.screen_recording.camera.pose_estimation.data.PoseSequence
import sensors_in_paradise.sonar.util.PreferencesHelper

class PoseSequenceViewHolder(
    private val context: Context,
    private val textureView: TextureView,
    onSourceLoadedListener: () -> Unit,
    onStartLoadingSource: () -> Unit,
    onSeekToNewPosition: ((ms: Long) -> Unit)? = null
) :
    VisualSequenceViewHolder(onSourceLoadedListener, onStartLoadingSource, onSeekToNewPosition) {
    private var poseSequences: List<ArrayList<List<PointF>>>? = null
    private var timeStampSequences: List<ArrayList<Long>>? = null
    private var jointsToDraw: List<Pair<Int, Int>>? = null
    private var startTime: Long? = null

    init {
        textureView.isOpaque = false
    }

    override fun loadSource(sourcePath: String, onSourceLoadedListener: () -> Unit) {
        val multiPoseSequence =
            PoseEstimationStorageManager.loadPoseSequenceFromCSV(context, sourcePath)

        setConvertedSequences(multiPoseSequence)

        jointsToDraw = VisualizationUtils.get2DLines(multiPoseSequence.type)
        startTime = multiPoseSequence.startTime

        onSourceLoadedListener()
    }

    /**
     *  Converts (time) series of 1 .. n poses, to n individual pose series' with own timestamps each.
     *  Sets sequence instance variables.
     */
    private fun setConvertedSequences(poseSequence: PoseSequence) {
        val numPoseInstances = poseSequence.posesArray.maxOf { it.size }

        val posesList = mutableListOf<ArrayList<List<PointF>>>()
        val timeStampsList = mutableListOf<ArrayList<Long>>()
        (0 until numPoseInstances).forEach { instanceIndex ->
            val poses =
                ArrayList(poseSequence.posesArray.mapNotNull { it.getOrNull(instanceIndex) })
            posesList.add(poses)

            val timeStamps = arrayListOf<Long>()
            poseSequence.timeStamps.filterIndexedTo(
                timeStamps
            ) { i, _ -> poseSequence.posesArray[i].getOrNull(instanceIndex) != null }
            timeStampsList.add(timeStamps)
        }
        poseSequences = posesList
        timeStampSequences = timeStampsList
    }

    override fun seekTo(ms: Long) {
        super.seekTo(ms)
        if (poseSequences != null) {
            val poses = poseSequences!!.indices.mapNotNull { getPoseAtTime(it, ms) }
            drawOnCanvas(poses)
        }
    }

    private fun getPoseAtTime(index: Int, ms: Long): List<PointF>? {
        val poseSequence = poseSequences!![index]
        val timeStampSequence = timeStampSequences!![index]

        val timeStamp = startTime!! + ms
        val seqIndex = timeStampSequence.binarySearch(timeStamp)
        return if (seqIndex < -1) {
            // When timeStamp lies between two samples
            VisualizationUtils.interpolatePose(
                poseSequence,
                timeStampSequence,
                -(seqIndex + 2),
                timeStamp
            )
        } else {
            poseSequence.getOrNull(seqIndex)?.map { PointF(it.x, it.y) }
        }
    }

    private fun drawOnCanvas(poses: List<List<PointF>>) {
        textureView.lockCanvas()?.let { canvas ->
            VisualizationUtils.transformPoints(
                poses,
                textureView.bitmap,
                canvas,
                VisualizationUtils.Transformation.PROJECT_ON_CANVAS
            )
            VisualizationUtils.drawSkeleton(
                poses,
                canvas,
                jointsToDraw!!,
                clearColor = if (PreferencesHelper.shouldShowPoseBackground(context)) null
                    else context.getColor(R.color.slightBackgroundContrast),
                circleColor = context.getColor(R.color.stickmanJoints),
                lineColor = context.getColor(R.color.backgroundContrast)
            )
            textureView.unlockCanvasAndPost(canvas)
        }
    }
}
