package sensors_in_paradise.sonar.util.dialogs

import android.app.AlertDialog
import android.app.Dialog
import android.content.Context
import android.content.DialogInterface
import android.view.LayoutInflater
import android.view.View

import android.widget.EditText
import android.widget.TextView
import androidx.core.widget.addTextChangedListener
import sensors_in_paradise.sonar.R

class TextInputDialog(
    context: Context,
    title: String,
    promptInterface: (text: String) -> Unit,
    hint: String = "",
    inputLabel: String = "Input your text here",
    startValue: String = "",
    errorMessage: String? = null,
    private val acceptanceInterface: ((text: String) -> Pair<Boolean, String?>)? = null
) {
    var dialog: AlertDialog

    init {
        val builder: AlertDialog.Builder = AlertDialog.Builder(context)
        builder.setTitle(title)

        val root = LayoutInflater.from(context).inflate(R.layout.prompt_dialog, null)
        val input = root.findViewById<EditText>(R.id.editText_promptDialog)
        val errorTV = root.findViewById<TextView>(R.id.tv_error_promptDialog)
        val inputLabelTV = root.findViewById<TextView>(R.id.tv_input_Label)
        if (errorMessage != null) {
            errorTV.text = errorMessage
        }
        inputLabelTV.text = inputLabel
        errorTV.visibility = if (errorMessage == null) View.GONE else View.VISIBLE
        input.hint = hint

        builder.setView(root)

        builder.setNegativeButton(
            "Cancel"
        ) { dialog, _ -> dialog.cancel() }

        dialog = builder.create()

        dialog.setButton(Dialog.BUTTON_POSITIVE, "OK") { _, _ ->
            promptInterface(input.text.toString())
        }

        dialog.show()
        input.addTextChangedListener { text ->
            val positiveBtn = dialog.getButton(AlertDialog.BUTTON_POSITIVE)

                val result = if (acceptanceInterface != null) acceptanceInterface!!(text.toString()) else Pair(true, "")
                positiveBtn.isEnabled = result.first
                if (result.second != null) {
                    errorTV.text = result.second
                }
                errorTV.visibility =
                    if (result.second != null && !result.first) View.VISIBLE else View.INVISIBLE
        }

        // Trigger change listener once
        input.setText(startValue)
    }

    fun setCancelListener(listener: DialogInterface.OnCancelListener) {
        dialog.setOnCancelListener(listener)
    }
}
