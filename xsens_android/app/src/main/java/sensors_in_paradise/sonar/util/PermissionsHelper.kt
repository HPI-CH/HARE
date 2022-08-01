package sensors_in_paradise.sonar.util

import android.content.Context
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import sensors_in_paradise.sonar.GlobalValues
import java.util.ArrayList

class PermissionsHelper private constructor() {
    companion object {
        fun areAllPermissionsGranted(context: Context): Boolean {
            val requiredPermissions = getRequiredButUngrantedPermissions(context)
            return requiredPermissions.size == 0
        }

        fun getRequiredButUngrantedPermissions(context: Context): ArrayList<String> {
            val result = ArrayList<String>()
            for (permission in GlobalValues.getRequiredPermissions()) {
                if (!isPermissionGranted(context, permission)) {
                    result.add(permission)
                }
            }
            return result
        }

        fun isPermissionGranted(context: Context, permission: String): Boolean {
            return (ContextCompat.checkSelfPermission(
                context,
                permission
            ) == PackageManager.PERMISSION_GRANTED)
        }
    }
}
