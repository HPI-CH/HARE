package sensors_in_paradise.sonar

import android.os.Handler
import android.os.Looper
import androidx.annotation.AnyThread
import java.util.concurrent.Executors

open class AsyncUI {
    private val uiHandler: Handler = Handler(Looper.getMainLooper())
    private val asyncExecutor = Executors.newFixedThreadPool(1)

    @AnyThread
    fun async(run: () -> Unit) {
        asyncExecutor.execute(run)
    }

    @AnyThread
    fun ui(runOnUiThread: () -> Unit) {
        uiHandler.post {
            runOnUiThread()
        }
    }
}
