package sensors_in_paradise.sonar

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.os.Build
import android.util.Log
import android.util.TypedValue
import androidx.annotation.AttrRes
import androidx.annotation.ColorInt
import androidx.core.content.ContextCompat
import com.xsens.dot.android.sdk.models.XsensDotPayload
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.net.NetworkInterface
import java.net.SocketException
import java.util.*
import kotlin.collections.ArrayList

class GlobalValues private constructor() {
    companion object {
        const val NULL_ACTIVITY = "null - activity"
        const val OTHERS_CATEGORY = "Others"
        const val UNKNOWN_PERSON = "unknown"
        const val ACTIVE_RECORDING_FLAG_FILENAME = "active"
        const val METADATA_JSON_FILENAME = "metadata.json"
        const val MEASUREMENT_MODE = XsensDotPayload.PAYLOAD_TYPE_CUSTOM_MODE_4

        fun getSensorRecordingsTempDir(context: Context): File {
            return context.dataDir.resolve("temp")
        }

        fun getVideoRecordingsTempDir(context: Context): File {
            return context.dataDir.resolve("videoTemp")
        }

        fun getPoseRecordingsTempDir(context: Context): File {
            return context.dataDir.resolve("poseTemp")
        }

        fun getRequiredPermissions(): ArrayList<String> {
            val result = arrayListOf(
                Manifest.permission.BLUETOOTH,
                Manifest.permission.BLUETOOTH_ADMIN,
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION,
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.INTERNET,
                Manifest.permission.ACCESS_NETWORK_STATE,
                Manifest.permission.CAMERA
            )
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                result.add(Manifest.permission.BLUETOOTH_SCAN)
                result.add(Manifest.permission.BLUETOOTH_CONNECT)
            }
            return result
        }

        fun formatTag(tagPrefix: String, deviceSetKey: String): String {
            return "$tagPrefix-$deviceSetKey"
        }

        fun getDurationAsString(durationMS: Long): String {

            val diffSecs = (durationMS) / 1000
            val minutes = diffSecs / 60
            val seconds = diffSecs - (diffSecs / 60) * 60

            return minutes.toString().padStart(2, '0') + ":" + seconds.toString().padStart(2, '0')
        }

        fun getMacAddress(): String {
            try {
                val all = Collections.list(NetworkInterface.getNetworkInterfaces())
                for (nif in all) {
                    if (!nif.name.equals("wlan0", ignoreCase=true)) continue

                    val macBytes = nif.hardwareAddress ?: return ""

                    val res1 = StringBuilder()
                    for (b in macBytes) {
                        res1.append(String.format(Locale.US, "%02X:", b))
                    }

                    if (res1.isNotEmpty()) {
                        res1.deleteCharAt(res1.length - 1)
                    }
                    return res1.toString()
                }
            } catch (ex: SocketException) {
                ex.message?.let { Log.e("GlobalValues", it) }
            }

            return "02:00:00:00:00:00"
        }

        private val fileEmojiMap = mapOf(
            "mp4" to "\uD83C\uDF9E️",
            "json" to "\uD83D\uDCD8",
            "csv" to "\uD83D\uDCCA"
        )

        fun getFileEmoji(file: File): String {
            if (file.isDirectory) {
                return "\uD83D\uDCC1"
            }
            val name = file.name
            val extension = name.substring(name.lastIndexOf(".") + 1)
            return fileEmojiMap[extension] ?: "\uD83D\uDCC4"
        }
        @Throws(NumberFormatException::class)
        fun getCSVHeaderAwareFileReader(inputFile: File): BufferedReader {
            val fileReader = BufferedReader(FileReader(inputFile))
            var line = fileReader.readLine()
            while (line != "") {
                line = fileReader.readLine()
            }
            return fileReader
        }

        fun getAndroidColorResource(context: Context, id: Int): Int {
            return context.getColorResCompat(id)
        }

        @ColorInt
        @SuppressLint("ResourceAsColor")
        private fun Context.getColorResCompat(@AttrRes id: Int): Int {
            val resolvedAttr = TypedValue()
            theme.resolveAttribute(id, resolvedAttr, true)
            val colorRes = resolvedAttr.run { if (resourceId != 0) resourceId else data }
            return ContextCompat.getColor(this, colorRes)
        }
    }
}
