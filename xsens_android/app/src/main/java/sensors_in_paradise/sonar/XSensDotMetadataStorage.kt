package sensors_in_paradise.sonar

import android.content.Context
import android.widget.Toast
import java.io.File

class XSensDotMetadataStorage(private val context: Context) :
    JSONStorage(File(context.dataDir, "xsensDotMetadata.json")) {

    override fun onFileNewlyCreated() {}

    override fun onJSONInitialized() {}

    fun setTagForAddress(address: String, name: String) {
        json.put(address, name)
        save()
    }

    fun getTagForAddress(address: String): String {
        return json.optString(address)
    }

    fun getAddressForTag(tag: String): String {
        json.keys().forEach {
            if (getTagForAddress(it) == tag) {
                return it
            }
        }
        return ""
    }

    fun tryGetDeviceSetKey(connectedDevices: XSENSArrayList): String? {
        val deviceSetKeys =
            connectedDevices.map { it.getSet() ?: "0" }
                .toSet()
        return if (deviceSetKeys.size == 1) {
            deviceSetKeys.first().toString()
        } else if (deviceSetKeys.isEmpty()) {
            Toast.makeText(context, "No device of known set connected!", Toast.LENGTH_SHORT).show()
            null
        } else {
            Toast.makeText(context, "Devices from multiple sets connected!", Toast.LENGTH_SHORT)
                .show()
            null
        }
    }
}
