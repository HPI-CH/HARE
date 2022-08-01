package sensors_in_paradise.sonar

import android.app.Activity
import android.os.Environment.DIRECTORY_PICTURES
import android.util.Log
import androidx.lifecycle.Lifecycle
import androidx.test.core.app.launchActivity
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry.getInstrumentation
import androidx.test.runner.screenshot.Screenshot
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import sensors_in_paradise.sonar.custom_views.confusion_matrix.ConfusionMatrix
import sensors_in_paradise.sonar.util.dialogs.ConfusionMatrixDialog
import java.io.IOException

@RunWith(AndroidJUnit4::class)
class GaitExperimentsTest {
    val instrumentation = getInstrumentation()

    @Before
    fun prepare() {
    }

    @Test
    fun testScreenshotTest() {
        launchActivity<MainActivity>().use { scenario ->

            scenario.onActivity { activity ->
                val matrices = listOf(
                    ConfusionMatrix(labels = arrayOf("Tired", "Not Tired")),
                    ConfusionMatrix(labels = arrayOf("2Tired", "2Not Tired"))
                )
                screenshotMatrices(activity, matrices)

                val dir =
                    android.os.Environment.getExternalStoragePublicDirectory(DIRECTORY_PICTURES)

                Log.d("GaitExperimentsTest", "$dir")
            }
            scenario.moveToState(Lifecycle.State.CREATED)
        }
    }

    private val TAG = "ScreenshotsUtils"
    fun screenshotMatrix(
        activity: Activity,
        matrices: List<ConfusionMatrix>,
        matrixIndex: Int,
        name: String,
        onDone: () -> Unit
    ) {
        val dialog = ConfusionMatrixDialog(activity, matrices, showAutomatically = false)
        dialog.setOnShowListener {
            Thread.sleep(1000)
            takeScreenshot(name)
            dialog.dismiss()

            onDone()
        }
        dialog.setDisplayConfusionMatrix(matrixIndex)
        dialog.show()
    }

    fun screenshotMatrices(
        activity: Activity,
        matrices: List<ConfusionMatrix>,
        matrixIndex: Int? = null
    ) {
        screenshotMatrix(activity, matrices, 0, "A") {
            screenshotMatrix(activity, matrices, 1, "B") {
            }
        }
    }

    fun takeScreenshot(screenShotName: String) {
        Log.d(TAG, "Taking screenshot of '$screenShotName'")

        val screenCapture = Screenshot.capture()
        try {
            screenCapture.apply {
                name = screenShotName
                process()
            }
            Log.d(TAG, "Screenshot taken")
        } catch (ex: IOException) {
            Log.e(TAG, "Could not take a screenshot", ex)
        }
    }
}
