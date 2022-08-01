package sensors_in_paradise.sonar.util.dialogs

import android.app.AlertDialog
import android.content.Context
import android.view.LayoutInflater
import android.widget.ProgressBar
import android.widget.TextView
import androidx.annotation.UiThread
import sensors_in_paradise.sonar.R

class ProgressDialog(context: Context) {
    private val progressBar: ProgressBar
    private val subprogressBar: ProgressBar
    private val progressLabelTV: TextView
    private val subprogressLabelTV: TextView
    private val progressTV: TextView
    private val subprogressTV: TextView
    private val dialog: AlertDialog
    init {
        val builder = AlertDialog.Builder(context)
        val root = LayoutInflater.from(context).inflate(R.layout.progress_dialog, null)

        progressBar = root.findViewById(R.id.progressBar_progress_progressDialog)
        subprogressBar = root.findViewById(R.id.progressBar_subprogress_progressDialog)

        progressLabelTV = root.findViewById(R.id.textView_progressLabel_progressDialog3)
        subprogressLabelTV = root.findViewById(R.id.textView_subprogressLabel_progressDialog)
        progressTV = root.findViewById(R.id.textView_progress_progressDialog2)
        subprogressTV = root.findViewById(R.id.textView_subprogress_progressDialog)

        builder.setView(root)
        builder.setCancelable(false)

        dialog = builder.create()
    }

    @UiThread
    fun setProgress(progress: Int, label: String? = null) {
        progressBar.progress = progress
        progressTV.text = "$progress%"
        if (label != null) {
            progressLabelTV.text = label
        }
    }

    @UiThread
    fun setSubProgress(progress: Int, label: String? = null) {
        subprogressBar.progress = progress
        subprogressTV.text = "$progress%"
        if (label != null) {
            subprogressLabelTV.text = label
        }
    }

    @UiThread
    fun dismiss() {
        dialog.dismiss()
    }

    @UiThread
    fun show() {
        dialog.show()
    }
}
