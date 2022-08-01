package sensors_in_paradise.sonar.util

import android.app.AlertDialog
import android.content.Context

class UIHelper private constructor() {
    companion object {
        fun showAlert(
            context: Context,
            message: String,
            title: String? = null,
            cancellable: Boolean = true
        ): AlertDialog {
            val builder = AlertDialog.Builder(context)
            builder.setCancelable(cancellable)
            builder.setMessage(message)
            if (title != null) {
                builder.setTitle(title)
            }
            val disconnectionAlertDialog = builder.create()
            disconnectionAlertDialog.show()
            return disconnectionAlertDialog
        }
    }
}
