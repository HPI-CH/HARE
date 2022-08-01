package sensors_in_paradise.sonar

import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import androidx.activity.result.ActivityResultLauncher
import sensors_in_paradise.sonar.util.PermissionsHelper
import sensors_in_paradise.sonar.util.UIHelper

class PermissionsHandler(private val requestPermissionLauncher: ActivityResultLauncher<Array<String>>) :
    ScreenInterface {
    lateinit var context: Context
    private var dialog: AlertDialog? = null

    override fun onActivityCreated(activity: Activity) {
        context = activity
        val missingPermissions = PermissionsHelper.getRequiredButUngrantedPermissions(context)
        requestPermissionLauncher.launch(missingPermissions.toArray(arrayOf<String>()))
    }

    override fun onActivityResumed() {
        dialog?.dismiss()
        if (!PermissionsHelper.areAllPermissionsGranted(context)) {
            val missingPermissions = PermissionsHelper.getRequiredButUngrantedPermissions(context)
            dialog = UIHelper.showAlert(
                context,
                missingPermissions.joinToString(separator = "\n\n"),
                title = "Permissions missing",
                cancellable = false
            )
        }
    }

    override fun onActivityWillDestroy() {
        // Nothing to do
    }
}
