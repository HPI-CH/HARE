package sensors_in_paradise.sonar

import android.content.Context
import android.util.Log
import android.widget.Toast
import androidx.annotation.UiThread
import com.xsens.dot.android.sdk.events.XsensDotData
import com.xsens.dot.android.sdk.interfaces.XsensDotMeasurementCallback
import com.xsens.dot.android.sdk.models.XsensDotDevice
import sensors_in_paradise.sonar.screen_connection.ConnectionInterface

class HeadingResetHandler(
    private val context: Context,
    private val scannedDevices: XSENSArrayList,
    private val onDeviceHeadingChanged: (address: String) -> Unit
) : XsensDotMeasurementCallback, ConnectionInterface {
    override fun onXsensDotHeadingChanged(deviceAdress: String?, status: Int, result: Int) {
        Log.d("Page1Handler", "onXsensDotHeadingChanged for device $deviceAdress")
        deviceAdress?.let {
            val device = scannedDevices[deviceAdress]
            device?.let {
                if (it.isResettingHeading || it.isRevertingHeading) {
                    it.stopMeasuring()
                    it.isResettingHeading = false
                    it.isRevertingHeading = false
                }
            }
            onDeviceHeadingChanged(deviceAdress)
        }
    }

    override fun onXsensDotRotLocalRead(deviceAdress: String?, quaternions: FloatArray?) {}

    override fun onConnectedDevicesChanged(deviceAddress: String, connected: Boolean) {
        if (connected) {
            scannedDevices[deviceAddress]?.setXsensDotMeasurementCallback(this)
        }
    }

    /* Resets the heading of all connected sensors by first starting real time streaming mode
    then resetting their headings once it started as required by the user manual
    * See section 4.1.3 https://www.xsens.com/hubfs/Downloads/Manuals/Xsens%20DOT%20User%20Manual.pdf*/
    @UiThread
    fun resetHeadings() {
        var count = 0
        val connectedDevices = scannedDevices.getConnectedWithOfflineMetadata()
        for (device in connectedDevices) {
            if (device.headingStatus != XsensDotDevice.HEADING_STATUS_XRM_HEADING) {
                device.isResettingHeading = true
                device.startMeasuring()
            } else {
                count++
            }
        }
        if (count > 0) {
            var msg = "$count/${connectedDevices.size} of connected devices already reset"
            if (count != connectedDevices.size) {
                msg += ". To align them all, revert them first"
            }
            Toast.makeText(
                context,
                msg,
                Toast.LENGTH_LONG
            ).show()
        }
    }

    /* Reverts the heading of all connected sensors by first starting real time streaming mode and
   then reverting their headings once it started as required by the user manual
   * See section 4.1.3 https://www.xsens.com/hubfs/Downloads/Manuals/Xsens%20DOT%20User%20Manual.pdf*/
    @UiThread
    fun revertHeadings() {
        var count = 0
        val connectedDevices = scannedDevices.getConnectedWithOfflineMetadata()
        for (device in scannedDevices.getConnectedWithOfflineMetadata()) {
            if (device.headingStatus == XsensDotDevice.HEADING_STATUS_XRM_HEADING) {
                device.isRevertingHeading = true
                device.startMeasuring()
            } else {
                count++
            }
        }
        if (count > 0) {
            Toast.makeText(
                context,
                "$count/${connectedDevices.size} of connected devices already reverted",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    override fun onXsensDotDataChanged(deviceAddress: String, xsensDotData: XsensDotData) {
        val device = scannedDevices[deviceAddress]
        if (device != null) {
            if (device.isResettingHeading) {
                device.resetHeading()
            }
            if (device.isRevertingHeading) {
                device.revertHeading()
            }
        }
    }
}
