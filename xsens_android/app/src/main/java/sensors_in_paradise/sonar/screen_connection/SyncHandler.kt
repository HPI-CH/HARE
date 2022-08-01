package sensors_in_paradise.sonar.screen_connection

import android.util.Log
import com.xsens.dot.android.sdk.interfaces.XsensDotSyncCallback
import java.util.HashMap

class SyncHandler(val callback: SyncInterface) : XsensDotSyncCallback {
    override fun onSyncingStarted(address: String?, isSuccess: Boolean, requestCode: Int) {
        Log.d("Sync", "onSyncingStarted address:" + address + " isSuccess: " + isSuccess)
    }

    override fun onSyncingProgress(progress: Int, requestCode: Int) {
        Log.d("Sync", "onSyncingProgress: " + progress)
        callback.onSyncProgress(progress)
    }

    override fun onSyncingResult(address: String?, isSuccess: Boolean, requestCode: Int) {
        callback.onFinishedSyncOfDevice(address, isSuccess)
        Log.d("Sync", "onSyncingResult: " + address + " isSuccess: " + isSuccess)
    }

    override fun onSyncingDone(syncingResultMap: HashMap<String, Boolean>?, isSuccess: Boolean, requestCode: Int) {
        Log.d("Sync", "onSyncingResult: isSuccess" + isSuccess)
        callback.onFinishedSyncing(syncingResultMap, isSuccess, requestCode)
    }

    override fun onSyncingStopped(address: String?, isSuccess: Boolean, requestCode: Int) {
        Log.d("Sync", "onSyncingStopped: isSuccess: " + isSuccess + " address:" + address)
    }
}
