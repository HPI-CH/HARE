package sensors_in_paradise.sonar

import com.xsens.dot.android.sdk.events.XsensDotData
import org.junit.Assert
import org.junit.Test
import sensors_in_paradise.sonar.machine_learning.InMemoryWindow

class InMemoryWindowTest {
    @Test
    fun initializationTest() {
        val features = arrayOf("Quat_Z_LF", "dq_W_LF", "dv[1]_LF").map { it.uppercase() }.toTypedArray()
        val window = InMemoryWindow(features, 2)
        assert(features.contentEquals(window.keys.toTypedArray()))
    }
    @Test
    fun addDataTest() {
        val features = arrayOf("Quat_Z_LF", "dq_W_RW", "dv[1]_LF")
        val window = InMemoryWindow(features, 2)
        for (feature in features) {
            assert(window.needsFeature(feature))
        }
        val data = XsensDotData().apply {
            quat = floatArrayOf(0f, 0f, 1f, 0f)
            dq = doubleArrayOf(0.0, 0.0, 0.2, 23.0)
            dv = doubleArrayOf(23.0, 0.0, 0.0)
        }

        window.appendSensorData("LF", data)
        Assert.assertFalse(window.hasEnoughDataToCompileWindow())
        window.appendSensorData("RW", data)
        Assert.assertFalse(window.hasEnoughDataToCompileWindow())
        data.sampleTimeFine += 1L

        window.appendSensorData("LF", data)
        Assert.assertFalse(window.hasEnoughDataToCompileWindow())
        data.sampleTimeFine += 1L

        window.appendSensorData("RW", data)
        Assert.assertTrue(window.hasEnoughDataToCompileWindow())

        window.compileWindow()
    }
    @Test
    fun forwardFillDataTest() {
        val features = arrayOf("Quat_W_LF", "Quat_X_LF", "Quat_Y_LF", "Quat_Z_LF")
        val window = InMemoryWindow(features, 3)

        val data0 = XsensDotData().apply {
            sampleTimeFine = 0L
            quat = floatArrayOf(0f, Float.NaN, 1f, 0f)
            dq = doubleArrayOf(0.0, 0.0, 0.2, 23.0)
            dv = doubleArrayOf(23.0, 0.0, Double.NaN)
            quat = floatArrayOf(0f, Float.NaN, 1f, 1f)
        }
        val data1 = XsensDotData().apply {
            sampleTimeFine = 1L
            quat = floatArrayOf(0f, 1f, Float.NaN, 2f)
        }
        val data2 = XsensDotData().apply {
            sampleTimeFine = 2L
            quat = floatArrayOf(0f, Float.NaN, 0f, 0f)
        }

        window.appendSensorData("LF", data0)
        Assert.assertFalse(windowHasNan(window))
        window.appendSensorData("LF", data1)
        Assert.assertFalse(windowHasNan(window))
        window.appendSensorData("LF", data2)
        Assert.assertFalse(windowHasNan(window))
        Assert.assertTrue(window["QUAT_W_LF"]!! == arrayListOf(Pair(0L, 0f), Pair(1L, 0f), Pair(2L, 0f)))
        Assert.assertTrue(window["QUAT_X_LF"]!! == arrayListOf(Pair(0L, 0f), Pair(1L, 1f), Pair(2L, 1f)))
        Assert.assertTrue(window["QUAT_Y_LF"]!! == arrayListOf(Pair(0L, 1f), Pair(1L, 1f), Pair(2L, 0f)))
        Assert.assertTrue(window["QUAT_Z_LF"]!! == arrayListOf(Pair(0L, 1f), Pair(1L, 2f), Pair(2L, 0f)))
    }

    private fun windowHasNan(window: InMemoryWindow): Boolean {
        return window.values.any { arrayList -> arrayList.any { it.second.isNaN() } }
    }
}
