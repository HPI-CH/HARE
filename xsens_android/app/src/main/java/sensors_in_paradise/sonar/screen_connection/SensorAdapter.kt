package sensors_in_paradise.sonar.screen_connection

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.drawable.Drawable
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Button
import android.widget.ProgressBar
import android.widget.ViewFlipper
import android.widget.ImageView
import androidx.appcompat.content.res.AppCompatResources
import androidx.cardview.widget.CardView
import androidx.recyclerview.widget.RecyclerView
import com.xsens.dot.android.sdk.models.XsensDotDevice
import sensors_in_paradise.sonar.R
import sensors_in_paradise.sonar.XSENSArrayList
import sensors_in_paradise.sonar.util.UIHelper

class SensorAdapter(
    context: Context,
    private val devices: XSENSArrayList,
    private val connectionCallbackUI: UIDeviceConnectionInterface
) :
    RecyclerView.Adapter<SensorAdapter.ViewHolder>() {
    private var disconnectedDrawable: Drawable? =
        AppCompatResources.getDrawable(context, R.drawable.ic_baseline_link_off_24)
    private var connectedDrawable: Drawable? =
        AppCompatResources.getDrawable(context, R.drawable.ic_baseline_link_24)
    private var syncedDrawable: Drawable? =
        AppCompatResources.getDrawable(context, R.drawable.ic_baseline_sync_24)

    /**
     * Provide a reference to the type of views that you are using
     * (custom ViewHolder).
     */
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val card: CardView = view.findViewById(R.id.card_sensorDevice)
        val nameTextView: TextView = view.findViewById(R.id.tv_name_sensorDevice)
        val detailsTextView: TextView = view.findViewById(R.id.tv_details_sensorDevice)
        val connectButton: Button = view.findViewById(R.id.switch_connect_sensorDevice)
        val flipper: ViewFlipper = view.findViewById(R.id.flipper_sensorDevice)
        val batteryPB: ProgressBar = view.findViewById(R.id.pb_battery_sensorDevice)
        val cancelButton: Button = view.findViewById(R.id.button_cancel_connection_sensor_device)
        val statusIV: ImageView = view.findViewById(R.id.imageView_status_connection_fragment)
        val sensorSetView: View = view.findViewById(R.id.view_sensorSetColor_sensorDevice)
        val tagSpellingWarningTextView: TextView =
            view.findViewById(R.id.tv_sensorTagSpellingWarning_sensorDevice)
    }

    // Create new views (invoked by the layout manager)
    override fun onCreateViewHolder(viewGroup: ViewGroup, viewType: Int): ViewHolder {
        // Create a new view, which defines the UI of the list item
        val view = LayoutInflater.from(viewGroup.context)
            .inflate(R.layout.sensor_device, viewGroup, false)

        return ViewHolder(view)
    }

    // Replace the contents of a view (invoked by the layout manager)
    @SuppressLint("SetTextI18n")
    override fun onBindViewHolder(viewHolder: ViewHolder, position: Int) {

        // Get element from your dataset at this position and replace the
        // contents of the view with that element

        val device = devices[position]
        viewHolder.nameTextView.text = "${device.name} ${device.tag}"

        viewHolder.connectButton.setOnClickListener {
            val isConnectedState = (viewHolder.connectButton.text == "Connect")
            connectionCallbackUI.onConnectionUpdateRequested(device, isConnectedState)
        }
        viewHolder.cancelButton.setOnClickListener {
            connectionCallbackUI.onConnectionCancelRequested(device)
        }
        viewHolder.card.setOnLongClickListener {
            device.identifyDevice()
        }
        val isConnected = (device.connectionState == XsensDotDevice.CONN_STATE_CONNECTED)
        val isConnecting = (device.connectionState == XsensDotDevice.CONN_STATE_CONNECTING)
        val isSynced = device.isSynced
        val isReconnecting = (device.connectionState == XsensDotDevice.CONN_STATE_RECONNECTING)
        val isConnectingOrReconnecting = isConnecting or isReconnecting
        viewHolder.flipper.displayedChild = if (isConnectingOrReconnecting) 1 else 0
        viewHolder.connectButton.text = if (isConnected) "Disconnect" else "Connect"
        var detailsText = getConnectionStateLabel(device.connectionState)
        if (isConnected) {
            detailsText = if (isSynced) "Synced" else "Connected"
            detailsText += " " + device.currentOutputRate + "Hz"
            detailsText += "\nHeading " + getHeadingStateLabel(device.headingStatus)
        }
        viewHolder.detailsTextView.text = detailsText
        viewHolder.batteryPB.progress = if (isConnected) device.batteryPercentage else 0
        viewHolder.batteryPB.visibility = if (isConnected) View.VISIBLE else View.GONE

        viewHolder.cancelButton.visibility =
            if (connectionCallbackUI.isSyncing) View.GONE else View.VISIBLE

        val hasSetColor = device.hasSetColor()
        if (hasSetColor) {
            viewHolder.sensorSetView.setBackgroundColor(device.getSetColor())
            viewHolder.sensorSetView.visibility = View.VISIBLE
        } else {
            viewHolder.sensorSetView.visibility = View.INVISIBLE
        }

        var statusDrawable = disconnectedDrawable
        if (isConnected) {
            statusDrawable = if (isSynced) syncedDrawable else connectedDrawable
        }

        viewHolder.statusIV.setImageDrawable(statusDrawable)

        val isSensorTagCompliant = device.isTagValid()
        viewHolder.tagSpellingWarningTextView.apply {
            visibility =
                if (isSensorTagCompliant || !isConnected) View.GONE else View.VISIBLE
            setOnClickListener {
                UIHelper.showAlert(
                    context,
                    context.getString(R.string.sensor_tag_prefix_pattern_explanation),
                    "Sensor tags not compliant"
                )
            }
        }
    }

    private fun getConnectionStateLabel(connectionState: Int): String {
        return when (connectionState) {
            XsensDotDevice.CONN_STATE_CONNECTED -> "Connected"
            XsensDotDevice.CONN_STATE_CONNECTING -> "Connecting"
            XsensDotDevice.CONN_STATE_DISCONNECTED -> "Disconnected"
            XsensDotDevice.CONN_STATE_RECONNECTING -> "Reconnecting"
            XsensDotDevice.CONN_STATE_START_RECONNECTING -> "Starting to reconnect"
            else -> "Unknown"
        }
    }

    private fun getHeadingStateLabel(headingState: Int): String {
        return when (headingState) {
            XsensDotDevice.HEADING_STATUS_XRM_DEFAULT_ALIGNMENT -> "default"
            XsensDotDevice.HEADING_STATUS_XRM_HEADING -> "reset"
            XsensDotDevice.HEADING_STATUS_XRM_NONE -> "default"
            else -> "unknown"
        }
    }

    // Return the size of your dataset (invoked by the layout manager)
    override fun getItemCount(): Int {
        return devices.size
    }

    fun notifyItemChanged(address: String) {
        val index = devices.indexOf(address)
        if (index != -1) {
            notifyItemChanged(index)
        }
    }
}
