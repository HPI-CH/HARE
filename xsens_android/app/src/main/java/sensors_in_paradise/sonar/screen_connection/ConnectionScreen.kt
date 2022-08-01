package sensors_in_paradise.sonar.screen_connection

import android.annotation.SuppressLint
import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.graphics.Color
import android.os.Build
import android.util.Log
import android.view.View
import android.widget.*
import androidx.annotation.RequiresApi
import androidx.recyclerview.widget.RecyclerView
import com.xsens.dot.android.sdk.XsensDotSdk
import com.xsens.dot.android.sdk.events.XsensDotData
import com.xsens.dot.android.sdk.interfaces.XsensDotDeviceCallback
import com.xsens.dot.android.sdk.interfaces.XsensDotScannerCallback
import com.xsens.dot.android.sdk.models.FilterProfileInfo
import com.xsens.dot.android.sdk.models.XsensDotDevice
import com.xsens.dot.android.sdk.models.XsensDotSyncManager
import com.xsens.dot.android.sdk.utils.XsensDotScanner
import sensors_in_paradise.sonar.util.PermissionsHelper
import java.util.ArrayList
import java.util.HashMap
import android.view.animation.AccelerateDecelerateInterpolator
import sensors_in_paradise.sonar.*
import kotlin.properties.Delegates

class ConnectionScreen(private val scannedDevices: XSENSArrayList) :
    XsensDotScannerCallback, XsensDotDeviceCallback, ScreenInterface,
    UIDeviceConnectionInterface, SyncInterface {
    private lateinit var context: Context
    private lateinit var activity: Activity
    private val connectionInterfaces = ArrayList<ConnectionInterface>()
    private lateinit var tv: TextView
    private lateinit var pb: ProgressBar
    private lateinit var rv: RecyclerView
    private lateinit var linearLayoutCenter: LinearLayout
    private lateinit var sensorAdapter: SensorAdapter
    private lateinit var syncBtn: Button
    private lateinit var refreshLinearLayout: LinearLayout
    private lateinit var refreshButton: Button
    private lateinit var xsensDotMetadata: XSensDotMetadataStorage
    private var unsyncedColor by Delegates.notNull<Int>()
    private val syncedColor = Color.parseColor("#00e676")
    override var isSyncing = false

    override fun onXsensDotConnectionChanged(address: String, state: Int) {
        activity.runOnUiThread {
            for (connectionInterface in connectionInterfaces) {
                connectionInterface.onConnectedDevicesChanged(address,
                state == XsensDotDevice.CONN_STATE_CONNECTED)
            }
            updateSyncButtonState()
            if (state != XsensDotDevice.CONN_STATE_CONNECTED) {
                sensorAdapter.notifyItemChanged(address)
            }
        }
    }

    override fun onXsensDotServicesDiscovered(address: String, status: Int) {
    }

    override fun onXsensDotFirmwareVersionRead(s: String, s1: String) {}

    override fun onXsensDotTagChanged(address: String, tag: String) {
        val device = scannedDevices[address]
        if (device != null) {
            Log.d("CONNECTION_SCREEN", "setting tag for device $address tag: $tag")
            xsensDotMetadata.setTagForAddress(address, tag)
        }
        activity.runOnUiThread {
            sensorAdapter.notifyItemChanged(address)
        }
    }

    override fun onXsensDotBatteryChanged(s: String, i: Int, i1: Int) {}

    override fun onXsensDotDataChanged(address: String, xsensDotData: XsensDotData) {
        for (connectionInterface in connectionInterfaces) {
            connectionInterface.onXsensDotDataChanged(address, xsensDotData)
        }
    }

    override fun onXsensDotInitDone(address: String) {
        activity.runOnUiThread {
            sensorAdapter.notifyItemChanged(address)
        }
    }

    override fun onXsensDotButtonClicked(s: String, l: Long) {}
    override fun onXsensDotPowerSavingTriggered(s: String) {}
    override fun onReadRemoteRssi(s: String, i: Int) {}
    override fun onXsensDotOutputRateUpdate(address: String, outputRate: Int) {
        for (connectionInterface in connectionInterfaces) {
            connectionInterface.onXsensDotOutputRateUpdate(address, outputRate)
        }
    }

    override fun onXsensDotFilterProfileUpdate(s: String, i: Int) {}
    override fun onXsensDotGetFilterProfileInfo(
        s: String,
        arrayList: ArrayList<FilterProfileInfo>
    ) {
    }

    override fun onSyncStatusUpdate(s: String, b: Boolean) {}
    override fun onXsensDotScanned(device: BluetoothDevice, i: Int) {
        if (!scannedDevices.contains(device.address)) {
            val tag = xsensDotMetadata.getTagForAddress(device.address)
            val index = scannedDevices.insertSortedBySet(XSensDotDeviceWithOfflineMetadata(context, device, this, tag))
            sensorAdapter.notifyItemInserted(index)
        }

        activity.runOnUiThread {
            linearLayoutCenter.visibility = View.INVISIBLE
            refreshLinearLayout.visibility = View.VISIBLE
        }
    }

    @SuppressLint("NotifyDataSetChanged")
    @RequiresApi(Build.VERSION_CODES.S)
    override fun onActivityCreated(activity: Activity) {
        this.context = activity
        this.activity = activity
        tv = activity.findViewById(R.id.tv_center_acitivity_main)
        rv = activity.findViewById(R.id.rv_bluetoothDevices_activity_main)
        pb = activity.findViewById(R.id.pb_activity_main)
        syncBtn = activity.findViewById(R.id.button_sync_connection_fragment)
        unsyncedColor = activity.getColor(R.color.colorAccent)

        linearLayoutCenter = activity.findViewById(R.id.linearLayout_center_activity_main)
        sensorAdapter = SensorAdapter(context, scannedDevices, this)
        rv.adapter = sensorAdapter
        refreshLinearLayout = activity.findViewById(R.id.linearLayout_refresh_connection_fragment)
        refreshButton = activity.findViewById(R.id.button_refresh_connection_fragment)
        xsensDotMetadata = XSensDotMetadataStorage(context)

        syncBtn.setOnClickListener { startSyncing() }

        refreshButton.setOnClickListener {
            val deg: Float = refreshButton.rotation + 1080f
            refreshButton.animate().rotation(deg).interpolator = AccelerateDecelerateInterpolator()
            refreshButton.animate().duration = 1500

            mXsScanner!!.stopScan()
            scannedDevices.forEach { it.disconnect() }
            scannedDevices.clear()
            sensorAdapter.notifyDataSetChanged()
            mXsScanner!!.startScan()
        }
    }

    @SuppressLint("SetTextI18n")
    override fun onActivityResumed() {
        if (PermissionsHelper.areAllPermissionsGranted(context)) {
            val mBluetoothAdapter = BluetoothAdapter.getDefaultAdapter()
            if (mBluetoothAdapter == null) {
                // Device does not support Bluetooth
                tv.text = "Scan started: false\nDevice does not support bluetooth"
                pb.visibility = View.INVISIBLE
                tv.visibility = View.VISIBLE
            } else if (!mBluetoothAdapter.isEnabled) {
                // Bluetooth is not enabled :)
                tv.text = "Scan started: false\nPlease enable bluetooth"
                pb.visibility = View.INVISIBLE
                tv.visibility = View.VISIBLE
            } else {
                // Bluetooth is enabled
                XsensDotSdk.setDebugEnabled(true)
                initXsScanner()
                val scanStarted = mXsScanner!!.startScan()
                tv.text = "Scan started: $scanStarted"
                pb.visibility = View.VISIBLE
            }
        }
    }

    override fun onActivityWillDestroy() {
        // Nothing to do
    }

    private var mXsScanner: XsensDotScanner? = null
    private fun initXsScanner() {
        mXsScanner = XsensDotScanner(context, this)
        mXsScanner!!.setScanMode(ScanSettings.SCAN_MODE_BALANCED)
    }

    override fun onConnectionUpdateRequested(
        device: XsensDotDevice,
        wantsConnection: Boolean
    ) {
        if (wantsConnection) {
            device.connect()
        } else {
            device.disconnect()
        }
        activity.runOnUiThread {
            sensorAdapter.notifyItemChanged(device.address)
        }
    }

    override fun onConnectionCancelRequested(device: XsensDotDevice) {
        device.disconnect()

        activity.runOnUiThread {
            sensorAdapter.notifyItemChanged(device.address)
        }
    }

    override fun onFinishedSyncOfDevice(address: String?, isSuccess: Boolean) {
        if (address != null) {
            activity.runOnUiThread {
                sensorAdapter.notifyItemChanged(address)
                Log.d("Sync", "Sync finished of device: $address $isSuccess")
            }
        }
    }

    private fun startSyncing() {
        isSyncing = true

        // Set the correct sensor settings
        for (device in scannedDevices.getConnected()) {
            device.setOutputRate(60)
        }

        // Try to find the ST sensor to use as root sensor
        val stSensor = scannedDevices.getConnected().find {
                sensor -> sensor.tag.startsWith("ST")
        }
        val selectedRootSensor = stSensor ?: scannedDevices.getConnected()[0]
        selectedRootSensor.isRootDevice = true

        XsensDotSyncManager.getInstance(SyncHandler(this)).startSyncing(scannedDevices.getConnected(), 0)
    }

    override fun onFinishedSyncing(
        syncingResultMap: HashMap<String, Boolean>?,
        isSuccess: Boolean,
        requestCode: Int
    ) {
        isSyncing = false
        activity.runOnUiThread {
            Toast.makeText(context, "Finished syncing, success: $isSuccess", Toast.LENGTH_LONG).show()
            updateSyncButtonState()
            setSyncProgress(100)
            if (syncingResultMap != null) {
                for (key in syncingResultMap.keys) {
                    sensorAdapter.notifyItemChanged(key)
                }
            }
        }
    }

    override fun onSyncProgress(progress: Int) {
        activity.runOnUiThread {
            setSyncProgress(progress)
        }
    }

    @SuppressLint("SetTextI18n")
    private fun setSyncProgress(progress: Int) {
        syncBtn.text = "Syncing connected sensors: $progress%"
    }

    private fun updateSyncButtonState() {
        val areDevicesConnected = scannedDevices.getConnected().size > 0
        activity.runOnUiThread {
            syncBtn.isEnabled = areDevicesConnected and !isSyncing
            syncBtn.visibility = if (areDevicesConnected or isSyncing) View.VISIBLE else View.GONE
            val bgColor = if (scannedDevices.areAllConnectedDevicesSynced()) syncedColor else unsyncedColor
            syncBtn.setBackgroundColor(bgColor)
            syncBtn.text = if (isSyncing) "Syncing connected sensors..." else "Sync connected sensors"
        }
    }

    fun addConnectionInterface(connectionInterface: ConnectionInterface) {
        connectionInterfaces.add(connectionInterface)
    }

	fun notifyItemChanged(address: String) {
        sensorAdapter.notifyItemChanged(address)
    }
}
