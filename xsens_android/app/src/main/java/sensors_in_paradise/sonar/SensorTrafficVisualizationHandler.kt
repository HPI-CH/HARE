package sensors_in_paradise.sonar

import android.app.Activity
import android.content.Context
import android.util.Log
import android.view.MenuItem
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.LinearLayout
import androidx.annotation.UiThread
import com.xsens.dot.android.sdk.events.XsensDotData
import com.xsens.dot.android.sdk.utils.XsensDotParser
import sensors_in_paradise.sonar.custom_views.SensorDataTrafficIndicatorView
import sensors_in_paradise.sonar.custom_views.stickman.Render3DView
import sensors_in_paradise.sonar.custom_views.stickman.object3d.Sensor3D
import sensors_in_paradise.sonar.screen_connection.ConnectionInterface

class SensorTrafficVisualizationHandler(
    val activity: Activity,
    private val scannedDevices: XSENSArrayList,
    private val indicator: SensorDataTrafficIndicatorView,
    private val orientationRenderViewsLL: LinearLayout,
    private val showOrientationVisualizationMi: MenuItem
) : ConnectionInterface {
    private var orientationLLExpanded = false
    private val context: Context = activity
    private var connectedSensorAddressIndexMap = mutableMapOf<String, Int>()
    private val orientationRenderViews = ArrayList<Render3DView>()
    private var lastRefreshTime = 0L

    override fun onConnectedDevicesChanged(deviceAddress: String, connected: Boolean) {
        connectedSensorAddressIndexMap.clear()
        val connectedDevices = scannedDevices.getConnectedWithOfflineMetadata()
        val tags = Array(connectedDevices.size) { i -> connectedDevices[i].tag }
        activity.runOnUiThread {
            indicator.setNumSensors(tags)
        }

        for ((i, device) in connectedDevices.withIndex()) {
            connectedSensorAddressIndexMap[device.address] = i
        }
        activity.runOnUiThread {
            initializeOrientationRenderViews(connectedDevices)
        }
        if (connectedDevices.isEmpty()) {
            setOrientationVisible(false)
            showOrientationVisualizationMi.isChecked = false
        }
        showOrientationVisualizationMi.isEnabled = !connectedDevices.isEmpty()
    }

    override fun onXsensDotDataChanged(deviceAddress: String, xsensDotData: XsensDotData) {
        val index = connectedSensorAddressIndexMap[deviceAddress]
        if (index != null) {
            activity.runOnUiThread {
                indicator.setSensorDataReceived(index)
            }

            if (orientationLLExpanded) {
                val eulerAngles = XsensDotParser.quaternion2Euler(xsensDotData.quat)
                activity.runOnUiThread {
                    orientationRenderViews[index].objects3DToDraw[0].apply {
                        resetToDefaultState()
                        rotateEuler(
                            eulerAngles[1].toFloat(),
                            eulerAngles[2].toFloat(),
                            eulerAngles[0].toFloat()
                        )
                    }
                }
                lastRefreshTime = System.currentTimeMillis()
            }
        } else {
            Log.e(
                "SensorTrafficIndicatorHandler",
                "Received data from sensor that is not registered as connected device"
            )
        }
    }

    override fun onXsensDotOutputRateUpdate(deviceAddress: String, outputRate: Int) {}

    fun setOrientationVisible(visible: Boolean) {
        orientationLLExpanded = visible
        if (orientationLLExpanded) {
            orientationRenderViewsLL.visibility = View.VISIBLE
        } else {
            orientationRenderViewsLL.visibility = View.GONE
        }
    }

    @UiThread
    private fun initializeOrientationRenderViews(connectedDevices: XSENSArrayList) {
        orientationRenderViewsLL.removeAllViews()
        orientationRenderViews.clear()

        for (i in 0 until connectedDevices.size) {
            val renderView = Render3DView(context)
            renderView.enableYRotation = false
            renderView.showFPS = true
            renderView.camera.eye.z = -5f
            renderView.layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, MATCH_PARENT, 1.0f)
            renderView.addObject3D(Sensor3D())
            orientationRenderViewsLL.addView(renderView)
            orientationRenderViews.add(renderView)
        }
    }
}
