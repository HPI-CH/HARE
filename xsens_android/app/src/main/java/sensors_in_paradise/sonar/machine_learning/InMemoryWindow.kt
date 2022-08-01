package sensors_in_paradise.sonar.machine_learning

import android.util.Log
import com.xsens.dot.android.sdk.events.XsensDotData
import java.nio.FloatBuffer
import java.util.Collections.max

class InMemoryWindow(featuresWithSensorTagPrefix: Array<String>, val windowSize: Int = 90) :
    LinkedHashMap<String, ArrayList<Pair<Long, Float>>>() {
    class SensorsOutOfSyncException(msg: String = "Sensors are out of sync") :
        IllegalStateException(msg)

    private val featuresPerSensorTagPrefix = LinkedHashMap<String, List<String>>()
    private val nansPerFeature = LinkedHashMap<String, Int>()
    init {
        for (feature in featuresWithSensorTagPrefix) {
            if (feature.isNotEmpty()) {
                put(feature.uppercase(), ArrayList(windowSize + cushion))
                nansPerFeature[feature.uppercase()] = 0
            }
        }
        for (deviceTagPrefix in extractDeviceTagPrefixes(this.keys)) {
            Log.w(
                "InMemoryWindow-init",
                "deviceTagPrefix in features: $deviceTagPrefix"
            )
            featuresPerSensorTagPrefix[deviceTagPrefix] =
                extractFeatures(deviceTagPrefix, this.keys)
        }
    }

    fun needsFeature(featureWithSensorTagPrefix: String): Boolean {
        return featureWithSensorTagPrefix.uppercase() in this
    }

    fun appendSensorData(featureWithSensorTagPrefix: String, value: Float, timeStamp: Long) {
        var v = value
        val featureValues = this[featureWithSensorTagPrefix.uppercase()]
        if (featureValues != null) {
            // insert the new value at the end of the list or if the timestamp of the new value
            // is lower than the largest timestamp, insert the new value at the correct position
            var insertionIndex = featureValues.size
            if (featureValues.isNotEmpty()) {
                if (featureValues.last().first > timeStamp) {
                    insertionIndex = featureValues.indexOfLast { it.first < timeStamp } + 1
                }
            }
            if (value.isNaN()) {
                val nansReceived = nansPerFeature[featureWithSensorTagPrefix]
                nansPerFeature[featureWithSensorTagPrefix] = nansReceived!! + 1
                v =
                    if (insertionIndex == 0) 0.0f else featureValues[insertionIndex - 1].second
            }
            featureValues.add(
                insertionIndex, Pair(
                    timeStamp,
                    v
                )
            )
        }
    }

    fun appendSensorData(deviceTagPrefix: String, xsensDotData: XsensDotData) {
        val timeStamp: Long = xsensDotData.sampleTimeFine
        if (deviceTagPrefix !in featuresPerSensorTagPrefix.keys) {
            Log.w(
                "InMemoryWindow-appendSensorData",
                "The device tag prefix $deviceTagPrefix for the given sensor data is not " +
                        "mentioned in the features that should be collected for this window."
            )
            return
        }
        for (featureTag in featuresPerSensorTagPrefix[deviceTagPrefix]!!) {
            val featureKey = "${featureTag}_$deviceTagPrefix"
            val featureValues = this[featureKey]
            if (featureValues != null) {
                // replace nan values with the forward fill method except for the first time stamp
                val value = getValueFromXsensDotData(featureTag, xsensDotData)
                appendSensorData(featureKey, value, timeStamp)
            }
        }
    }

    private fun getValueFromXsensDotData(featureTag: String, xsensDotData: XsensDotData): Float {
        // Features can have tags with axis chars like Mag_Y, Quat_Z or dq_X
        // OR they can have index spelling like dv[1], dv[2] to determine their axis...
        val hasAxisIndexSpelling = featureTag.contains(indexSpellingRegex)
        val featureType =
            if (hasAxisIndexSpelling) featureTag.substringBefore("[") else featureTag.split("_")[0]
        val featureAxisTag = if (hasAxisIndexSpelling) featureTag.substringAfter("[")
            .substringBefore("]") else featureTag.split("_")[1]
        val featureAxisIndex = featureAxisTagToIndex(featureType, featureAxisTag)
        return when (featureType) {
            "QUAT" -> xsensDotData.quat[featureAxisIndex]
            "ACC" -> xsensDotData.acc[featureAxisIndex].toFloat()
            "MAG" -> xsensDotData.mag[featureAxisIndex].toFloat()
            "GYR" -> xsensDotData.gyr[featureAxisIndex].toFloat()
            "DQ" -> xsensDotData.dq[featureAxisIndex].toFloat()
            "DV" -> xsensDotData.dv[featureAxisIndex].toFloat()
            else -> throw IllegalArgumentException("Unknown feature type: $featureType")
        }
    }

    private fun featureAxisTagToIndex(featureType: String, featureAxisTag: String): Int {
        return when (featureAxisTag) {
            "W" -> 0
            "X" -> 1
            "Y" -> 2
            "Z" -> 3
            "1" -> 1
            "2" -> 2
            "3" -> 3
            else -> throw IllegalArgumentException("Invalid feature axis $featureAxisTag")
        } - if (featureType == "QUAT" || featureType == "DQ") 0 else 1
    }

    private fun extractDeviceTagPrefixes(keys: MutableSet<String>): List<String> {
        return keys.map { it.substringAfterLast("_") }.distinct()
            .toList()
    }

    private fun extractFeatures(deviceTagPrefix: String, keys: MutableSet<String>): List<String> {
        return keys.filter { it.endsWith(deviceTagPrefix) }.map { it.substringBeforeLast("_") }
            .toList()
    }

    fun getRequiredSensorTagPrefixes(): List<String> {
        return extractDeviceTagPrefixes(this.keys)
    }

    fun compileWindow(): FloatBuffer {
        // find starting indices of feature vectors closest to the starting timestamp
        val startingIndices = getStartIndices()

        // return byteBuffer of feature vectors starting at the starting indices
        val buffer = FloatBuffer.allocate(this.keys.size * windowSize)

        for (i in 0 until windowSize) {
            for ((j, key) in this.keys.withIndex()) {
                val index = startingIndices[j] + i
                if (index < this[key]!!.size) {
                    buffer.put(this[key]!![index].second)
                } else {
                    throw IllegalStateException("Not enough data to fill window for feature: $key")
                }
            }
        }
        buffer.rewind()
        return buffer
    }
    fun compileWindowToArray(): Array<FloatArray> {
        return Array(windowSize) { FloatArray(keys.size) }.apply { compileWindowToArray(this) }
    }
    fun compileWindowToArray(array: Array<FloatArray>) {
        // find starting indices of feature vectors closest to the starting timestamp
        val startingIndices = getStartIndices()

        // return array of feature vectors starting at the starting indices

        for (i in 0 until windowSize) {
            for ((j, key) in this.keys.withIndex()) {
                val index = startingIndices[j] + i
                if (index < this[key]!!.size) {
                    array[i][j] = this[key]!![index].second
                } else {
                    throw IllegalStateException("Not enough data to fill window for feature: $key")
                }
            }
        }
    }

    fun clearValues() {
        for (key in this.keys) {
            this[key]?.clear()
        }
    }

    @Throws(SensorsOutOfSyncException::class)
    private fun getStartIndices(): IntArray {
        // find latest starting timestamp of each feature
        val startingTimestamp: Long = max(this.values.map { if (it.isEmpty()) 0 else it[0].first })
        // find starting indices of feature vectors closest to the starting timestamp
        return this.values.map {
            val closestIndex = findClosestIndex(startingTimestamp, it)
            if (closestIndex > sensorsOutOfSyncMinStartIndexDifference) {
                throw SensorsOutOfSyncException("Sensors are out of synch by at least $closestIndex measurements")
            }
            return@map closestIndex
        }.toIntArray()
    }

    private fun hasEnoughDataToCompileWindow(startIndices: IntArray): Boolean {
        for ((i, key) in this.keys.withIndex()) {

            if (startIndices[i] + windowSize > (this[key]?.size ?: 0)) {
                return false
            }
        }
        return true
    }

    @Throws(SensorsOutOfSyncException::class)
    fun hasEnoughDataToCompileWindow(): Boolean {
        return hasEnoughDataToCompileWindow(getStartIndices())
    }

    private fun findClosestIndex(
        startingTimestamp: Long,
        valuesSorted: ArrayList<Pair<Long, Float>>
    ): Int {
        var closestIndex = 0
        var closestDifference = Long.MAX_VALUE
        for (i in 0 until valuesSorted.size) {
            val difference = Math.abs(valuesSorted[i].first - startingTimestamp)
            if (difference <= closestDifference) {
                closestIndex = i
                closestDifference = difference
            } else {
                return closestIndex
            }
        }
        return closestIndex
    }

    fun getNansReceivedPerFeature(): Map<String, Int> {
        return nansPerFeature
    }

    companion object {
        private const val cushion = 10
        private val indexSpellingRegex = Regex("\\[\\d\\]")
        private const val sensorsOutOfSyncMinStartIndexDifference = 100
    }
}
