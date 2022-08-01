package sensors_in_paradise.sonar.screen_recording.camera.pose_estimation

import android.content.Context
import android.os.Looper
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import autovalue.shaded.com.`google$`.common.math.`$IntMath`.pow
import org.jetbrains.kotlinx.dataframe.*
import org.jetbrains.kotlinx.dataframe.api.*
import org.jetbrains.kotlinx.dataframe.columns.ColumnSet
import org.jetbrains.kotlinx.dataframe.io.ColType
import org.jetbrains.kotlinx.dataframe.io.readCSV
import sensors_in_paradise.sonar.screen_recording.Recording
import sensors_in_paradise.sonar.screen_recording.RecordingMetadataStorage
import sensors_in_paradise.sonar.screen_recording.camera.pose_estimation.data.Pose
import java.io.FileNotFoundException
import java.util.*
import java.util.Collections.min
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.math.pow
import kotlin.math.abs
import kotlin.math.min

@Suppress("LongParameterList")
class SensorPlacementEstimator(
    val context: Context,
    estimatePlacementButton: Button,
    lessPositionsButton: Button,
    morePositionsButton: Button,
    private val numPositionsTV: TextView,
    private val numRecordingsTV: TextView,
    val onRunningToggle: ((Boolean) -> Unit),
    val onSelectedRecordingsChanged: ((List<Recording>) -> Unit)
) {
    val recordings = mutableListOf<Recording>()
    private var numPositions = 1

    private var isRunning = false

    companion object {
        val POSITIONS = SensorPlacementDialog.POSITIONS_MAP.keys
        const val MIN_FRAME_LENGTH = 500
        const val MIN_ESTIMATED_DURATION = 60
    }

    init {
        estimatePlacementButton.setOnClickListener {
            if (areRecordingsEligible()) {
                Thread {
                    Looper.prepare()
                    toggleRunning()
                    val scores = estimateSensorPlacements(recordings.toList(), numPositions)
                    scores?.let { SensorPlacementDialog(context).updateScores(it) }
                    toggleRunning()
                    Looper.loop()
                }.start()
            }
        }
        lessPositionsButton.setOnClickListener {
            numPositions = max(numPositions - 1, 1)
            onNumPositionsChanged()
        }
        morePositionsButton.setOnClickListener {
            numPositions = min(numPositions + 1, POSITIONS.size - 1)
            onNumPositionsChanged()
        }
        onNumPositionsChanged()
    }

    private fun toggleRunning() {
        isRunning = !isRunning
        onRunningToggle(isRunning)
    }

    private fun onNumPositionsChanged() {
        numPositionsTV.text = "$numPositions Sensor" + if (numPositions > 1) "s" else ""
    }

    /**
     * Returns [true] if recording is being added to selected recordings, else [false]
     */
    fun toggleRecordingSelection(recording: Recording): Boolean {
        return if (recording in recordings) {
            removeRecording(recording)
            false
        } else {
            tryAddRecording(recording)
        }
    }

    private fun removeRecording(recording: Recording) {
        if (isRunning && recordings.size == 1) {
            return
        }
        recordings.remove(recording)
        onRecordingsUpdate()
    }

    private fun tryAddRecording(recording: Recording): Boolean {
        if (!isRecordingEligible(recording)) {
            return false
        }
        recordings.add(recording)
        onRecordingsUpdate()
        return true
    }

    private fun onRecordingsUpdate() {
        onSelectedRecordingsChanged(recordings)
        numRecordingsTV.text = "${recordings.size}"
    }

    /**
     * Checks whether a recording contains valuable data at all
     */
    private fun isRecordingEligible(recording: Recording): Boolean {
        try {
            if (!recording.hasPoseSequenceRecording()) {
                throw FileNotFoundException("Placement Estimation requires Body Pose Sequence.")
            }

            val poseFilePath = recording.getPoseSequenceFile().absolutePath
            if (PoseEstimationStorageManager.getPoseTypeFromCSV(context, poseFilePath)
                != Pose.BodyPose
            ) {
                throw FileNotFoundException("Placement Estimation requires 'Body' Pose Sequence.")
            }
        } catch (e: FileNotFoundException) {
            Toast.makeText(
                context,
                e.message,
                Toast.LENGTH_SHORT
            ).show()
            return false
        }
        return true
    }

    /**
     * Checks efficiently whether the selected recordings are likely to suffice the required amount of data
     */
    private fun areRecordingsEligible(): Boolean {
        try {
            val estimatedActivityTimes = mutableMapOf<String, Long>()
            recordings.forEach { recording ->
                val activities = recording.metadataStorage.getActivities()
                activities.forEachIndexed { index, labelEntry ->
                    val activityEndTime = activities.getOrNull(index + 1)?.timeStarted
                        ?: recording.metadataStorage.getTimeEnded()
                    val duration = activityEndTime - labelEntry.timeStarted

                    val oldDuration = estimatedActivityTimes.getOrDefault(labelEntry.activity, 0L)
                    estimatedActivityTimes[labelEntry.activity] = oldDuration + (duration / 1000)
                }
            }
            val filteredActivityTimes = estimatedActivityTimes.filter { (activity, duration) ->
                duration >= MIN_ESTIMATED_DURATION && "null" !in activity
            }

            if (filteredActivityTimes.size < 2) {
                throw FileNotFoundException("Requires 2 Activities (not null) captured 1 min each.")
            }
        } catch (e: FileNotFoundException) {
            Toast.makeText(
                context,
                e.message,
                Toast.LENGTH_LONG
            ).show()
            return false
        }
        return true
    }

    @Suppress("TooGenericExceptionCaught")
    private fun estimateSensorPlacements(
        lockedRecordings: List<Recording>,
        lockedNumPositions: Int
    ): Map<List<String>, Float>? {
        try {
            var (df, activities) = createMergedDataFrame(lockedRecordings)

            df = mergeColumns(df)

            df = appendActivities(df, activities)

            df = centralizePose(df)

            val activityFrames = getFramesPerActivity(df).filter { (activity, _) ->
                "null" !in activity
            }

            if (activityFrames.size < 2) {
                throw FileNotFoundException("Not enough data provided.")
            }

            val positionSubsets = getAllSubsetOfSize(lockedNumPositions, POSITIONS.toList())
            val scores = positionSubsets.associateWith { positions ->
                calcPositionsScore(positions, activityFrames.values.toList())
            }
            return scores
        } catch (e: Exception) {
            Toast.makeText(
                context,
                "Estimation Failed:\n ${e.message}",
                Toast.LENGTH_LONG
            ).show()
            return null
        }
    }

    private fun calcScore(df1: DataColumn<Double>, df2: DataColumn<Double>): Float {
        val v1 = df1.asSequence()
        val v2 = df2.asSequence()

        val dotProduct = v1.zip(v2, Double::times).sum()
        val norm1 = sqrt(v1.map { it.pow(2) }.sum())
        val norm2 = sqrt(v2.map { it.pow(2) }.sum())

        val cosine = dotProduct / (norm1 * norm2)
        return abs(1f - cosine.toFloat())
    }

    private fun getMergedPositionsFrame(
        df: DataFrame<*>,
        positions: List<String>
    ): Pair<DataColumn<Double>, DataColumn<Double>> {
        var xColumn: DataColumn<Double>? = null
        var yColumn: DataColumn<Double>? = null
        positions.forEach { position ->
            val newXColumn = df["${position}_X"].convertToDouble().dropNulls()
            xColumn = xColumn?.concat(newXColumn) ?: newXColumn

            val newYColumn = df["${position}_Y"].convertToDouble().dropNulls()
            yColumn = yColumn?.concat(newYColumn) ?: newYColumn
        }
        return Pair(xColumn!!, yColumn!!)
    }

    private fun calcPositionsScore(
        positions: List<String>,
        activityFrames: List<DataFrame<*>>
    ): Float {
        var sensorScore = 0f
        var actIterator = 1
        activityFrames.forEach { frame1 ->
            val (x1, y1) = getMergedPositionsFrame(frame1, positions)

            for (i in activityFrames.size - 1 downTo actIterator) {
                val frame2 = activityFrames[i]
                val (x2, y2) = getMergedPositionsFrame(frame2, positions)

                sensorScore += calcScore(x1, x2)
                sensorScore += calcScore(y1, y2)
            }

            actIterator += 1
        }
        return sensorScore
    }

    private fun getAllSubsetOfSize(size: Int, fromList: List<String>): Set<List<String>> {
        val n = fromList.size
        if (size < 1 || size >= n) {
            return setOf()
        }

        val maxIt = pow(2, n)
        val subsets = mutableSetOf<List<String>>()
        for (i in 0 until maxIt) {
            var bitstring = i.toString(2)
            bitstring = "0".repeat(n - bitstring.length) + bitstring
            val indices = bitstring.toList().mapIndexedNotNull { index, value ->
                if (value == '1') index
                else null
            }
            if (indices.size == size) {
                subsets.add(indices.map { fromList[it] })
            }
        }
        return subsets
    }

    private fun getFramesPerActivity(df: DataFrame<*>): Map<String, DataFrame<Any?>> {
        val activityGroups = df.groupBy("Activity")
        val activityFrames = activityGroups.keys.values().mapIndexedNotNull { index, activity ->
            val frame = activityGroups.groups[index]
            if (frame.size().nrow >= MIN_FRAME_LENGTH) {
                activity.toString() to frame
            } else null
        }.toMap()

        // Cut Frames into equal size, by taking only the first minFrameSize rows
        val minFrameSize = min(activityFrames.map { (_, frame) -> frame.size().nrow })
        return activityFrames.mapValues { (_, frame) ->
            frame.take(minFrameSize)
        }
    }

    private fun centralizePose(df: DataFrame<*>): DataFrame<*> {
        val xColumns = df.getColumns { endsWith("_X") }.toColumnGroup("xColumns")
        var newDf = df.update { endsWith("_X") }.perRowCol { row, col ->
            val massPoint = row[xColumns].rowMean()
            val translation = massPoint - 0.5
            (row[col] as Double) - translation
        }
        val yColumns = df.getColumns { endsWith("_Y") }.toColumnGroup("yColumns")
        newDf = newDf.update { endsWith("_Y") }.perRowCol { row, col ->
            val massPoint = row[yColumns].rowMean()
            val translation = massPoint - 0.5
            (row[col] as Double) - translation
        }
        return newDf
    }

    private fun appendActivities(
        df: DataFrame<*>,
        activities: ArrayList<RecordingMetadataStorage.LabelEntry>
    ): DataFrame<*> {
        var iterator = 0
        val activityColumn = df.mapToColumn("Activity") { row ->
            val ts = (row["TimeStamp"] as Long)
            if (iterator == activities.size - 1 || ts < activities[iterator + 1].timeStarted) {
                activities[iterator].activity
            } else {
                iterator += 1
                activities[iterator].activity
            }
        }
        return df.plus(activityColumn)
    }

    private fun mergeColumns(df: DataFrame<*>): DataFrame<*> {
        var mergedDf = df

        // Merge Head Points
        mergedDf = mergedDf.merge {
            ("NOSE_X" and "LEFT_EYE_X" and "RIGHT_EYE_X" and "LEFT_EAR_X" and "RIGHT_EAR_X") as ColumnSet<Double>
        }.by { it.average() }.into("HEAD_X")
        mergedDf = mergedDf.merge {
            ("NOSE_Y" and "LEFT_EYE_Y" and "RIGHT_EYE_Y" and "LEFT_EAR_Y" and "RIGHT_EAR_Y") as ColumnSet<Double>
        }.by { it.average() }.into("HEAD_Y")

        // Merge Hip Points
        mergedDf = mergedDf.merge { ("LEFT_HIP_X" and "RIGHT_HIP_X") as ColumnSet<Double> }
            .by { it.average() }.into("HIP_X")
        mergedDf = mergedDf.merge { ("LEFT_HIP_Y" and "RIGHT_HIP_Y") as ColumnSet<Double> }
            .by { it.average() }.into("HIP_Y")

        return mergedDf
    }

    private fun appendPoseFileToDataFrame(
        df: DataFrame<*>?,
        poseFilePath: String,
        colTypes: Map<String, ColType>
    ): DataFrame<*> {
        val newDf = DataFrame.readCSV(
            poseFilePath,
            colTypes = colTypes,
            delimiter = ',',
            skipLines = PoseEstimationStorageManager.getHeaderLineSize(poseFilePath),
            parserOptions = ParserOptions(Locale.ENGLISH)
        )
        return df?.concat(newDf) ?: newDf
    }

    private fun createMergedDataFrame(
        lockedRecordings: List<Recording>
    ): Pair<DataFrame<*>, ArrayList<RecordingMetadataStorage.LabelEntry>> {
        val colTypes =
            PoseEstimationStorageManager.getCSVColumns(Pose.BodyPose).associateWith { colName ->
                when (colName) {
                    "TimeStamp" -> ColType.Long
                    else -> ColType.Double
                }
            }

        var df: DataFrame<*>? = null
        val activities = ArrayList<RecordingMetadataStorage.LabelEntry>()
        val sortedRecordings = lockedRecordings.sortedBy { r -> r.metadataStorage.getTimeStarted() }
        sortedRecordings.forEach { recording ->
            val poseFilePath = recording.getPoseSequenceFile().absolutePath
            df = appendPoseFileToDataFrame(df, poseFilePath, colTypes)
            activities.addAll(recording.metadataStorage.getActivities())
        }
        return Pair(df!!, activities)
    }
}
