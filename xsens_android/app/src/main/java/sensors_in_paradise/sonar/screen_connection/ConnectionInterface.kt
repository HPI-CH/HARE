package sensors_in_paradise.sonar.screen_connection

import androidx.annotation.AnyThread
import androidx.annotation.UiThread
import com.xsens.dot.android.sdk.events.XsensDotData

interface ConnectionInterface {
    @UiThread
    fun onConnectedDevicesChanged(deviceAddress: String, connected: Boolean)
    @AnyThread
    fun onXsensDotDataChanged(deviceAddress: String, xsensDotData: XsensDotData) {}
    @AnyThread
    fun onXsensDotOutputRateUpdate(deviceAddress: String, outputRate: Int) {}
}
