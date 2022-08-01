package sensors_in_paradise.sonar.util.dialogs

import android.app.AlertDialog
import android.app.Dialog
import android.content.Context

class SingleChoiceDialog(
    context: Context,
    title: String,
    options: Array<String>,
    private var checkedItem: Int = -1,
    onItemChosen: (item: String) -> Unit,
    neutralButtonText: String? = null,
    neutralButtonListener: ((dialog: AlertDialog) -> Unit)? = null,
) {

    init {
        val builder = AlertDialog.Builder(context)
        builder.setTitle(title)
        builder.setSingleChoiceItems(options, checkedItem) { _, which -> checkedItem = which }
        builder.setPositiveButton("OK") { _, _ ->
            if (checkedItem != -1) {
                onItemChosen(options[checkedItem])
            }
        }
        builder.setNegativeButton("Cancel", null)

        if (neutralButtonText != null) {
            builder.setNeutralButton(neutralButtonText, null)
        }
        val dialog = builder.create()
        dialog.setOnShowListener {
            if (neutralButtonText != null) {
                dialog.getButton(Dialog.BUTTON_NEUTRAL).setOnClickListener { _ ->
                    neutralButtonListener?.invoke(dialog)
                }
            }
        }
        dialog.show()
    }
}
