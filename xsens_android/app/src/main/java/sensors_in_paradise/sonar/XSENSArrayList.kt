package sensors_in_paradise.sonar

import com.xsens.dot.android.sdk.models.XsensDotDevice

class XSENSArrayList : ArrayList<XSensDotDeviceWithOfflineMetadata>() {
    operator fun contains(deviceAddress: String): Boolean {

        for (device in this) {
            if (device.address == deviceAddress) {
                return true
            }
        }
        return false
    }

    operator fun get(deviceAddress: String): XSensDotDeviceWithOfflineMetadata? {
        for (device in this) {
            if (device.address == deviceAddress) {
                return device
            }
        }
        return null
    }

	fun indexOf(deviceAddress: String): Int {
        for ((index, device) in this.withIndex()) {
            if (device.address == deviceAddress) {
                return index
            }
        }
        return -1
    }

	fun getConnected(): ArrayList<XsensDotDevice> {
        val devices = ArrayList<XsensDotDevice>()
        for (device in this) {
            if (device.connectionState == XsensDotDevice.CONN_STATE_CONNECTED) {
                devices.add(device)
            }
        }
        return devices
    }

	fun getConnectedWithOfflineMetadata(): XSENSArrayList {
        val devices = XSENSArrayList()
        for (device in this) {
            if (device.connectionState == XsensDotDevice.CONN_STATE_CONNECTED) {
                devices.add(device)
            }
        }
        return devices
    }

	fun areAllConnectedDevicesSynced(): Boolean {
        for (device in getConnected()) {
            if (!device.isSynced) {
                return false
            }
        }
        return true
    }

	fun insertSortedBySet(device: XSensDotDeviceWithOfflineMetadata): Int {
        val set = device.getSet()
        for (i in 0 until size) {
            var returnIndex = -1
            val currentSet = this[i].getSet()
            if (set != null) {
                if (currentSet != null) {
                    // > 0 if currentSet < set
                    if (set <= currentSet) {
                        returnIndex = i
                    }
                } else {
                    returnIndex = i
                }
            } else if (currentSet == null) {
                returnIndex = i
            }
            if (returnIndex != -1) {
                add(returnIndex, device)
                return returnIndex
            }
        }
        add(device)
        return size - 1
    }
}
