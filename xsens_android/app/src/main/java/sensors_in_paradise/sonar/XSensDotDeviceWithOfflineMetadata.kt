package sensors_in_paradise.sonar

import android.bluetooth.BluetoothDevice
import android.content.Context
import android.graphics.Color
import com.xsens.dot.android.sdk.interfaces.XsensDotDeviceCallback
import com.xsens.dot.android.sdk.models.XsensDotDevice
import java.util.regex.Pattern

class XSensDotDeviceWithOfflineMetadata(
    context: Context?,
    device: BluetoothDevice?,
    cb: XsensDotDeviceCallback?,
    private val _tag: String?
) : XsensDotDevice(context, device, cb) {
    private val defaultTag = "Xsens DOT"
    var isResettingHeading = false
    var isRevertingHeading = false

    override fun getTag(): String {
        val shouldUseOfflineTag =
            ((super.getTag() == "" || super.getTag() == defaultTag) && _tag != null)
        return if (shouldUseOfflineTag) _tag!! else super.getTag()
    }

	fun getSet(): String? {
        return extractDeviceSetKeyFromTag(tag)
    }

	fun getSetColor(): Int {
        return Color.parseColor(
            when (getSet()) {
                "1" -> "#ff9e80"
                "2" -> "#b9f6ca"
                "3" -> "#ea80fc"
                else -> "#263238"
            }
        )
    }
    fun getTagPrefix(): String? {
        return extractTagPrefixFromTag(tag)
    }

    fun isTagValid(): Boolean {
        return tag.matches(sensorTagRegex)
    }

    fun hasSetColor(): Boolean {
        return when (getSet()) {
            "1", "2", "3" -> true
            else -> false
        }
    }

    companion object {
        fun extractDeviceSetKeyFromTag(tag: String): String? {
            return if (doesTagMatchPattern(tag)) tag.last().toString() else null
        }
        private val regex = Pattern.compile("(LF|LW|ST|RW|RF)-\\d")
        private fun doesTagMatchPattern(tag: String): Boolean {
            return regex.matcher(tag).matches()
        }
        fun extractTagPrefixFromTag(tag: String): String? {
            return if (doesTagMatchPattern(tag)) tag.substring(0, 2) else null
        }
        private val sensorTagRegex = Regex("^[A-Z]+-\\d+\$")
    }
}
