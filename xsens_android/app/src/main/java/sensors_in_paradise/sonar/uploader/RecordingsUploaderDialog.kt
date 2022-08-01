package sensors_in_paradise.sonar.uploader

import android.annotation.SuppressLint
import android.app.Activity
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import android.widget.ViewSwitcher
import androidx.appcompat.app.AlertDialog
import androidx.recyclerview.widget.RecyclerView
import sensors_in_paradise.sonar.R
import sensors_in_paradise.sonar.util.PreferencesHelper

@SuppressLint("NotifyDataSetChanged", "SetTextI18n")
class RecordingsUploaderDialog(activity: Activity, uploader: DavCloudRecordingsUploader) :
    AlertDialog(activity) {
    val context = activity

    private val recordings = uploader.recordingUiItems
    private val adapter = RecordingsUploadAdapter(recordings)
    private var hintTV: TextView
    private var noFilesTV: TextView
    private var endPointDetailsTV: TextView
    private var switcher: ViewSwitcher
    private var uploadButton: Button
    private var activityCountTextView: TextView
    private val recordingsManager = uploader.recordingsManager

    init {
        recordingsManager.reloadRecordingsFromStorage()
        val inflater = activity.layoutInflater
        val rootView = inflater.inflate(R.layout.upload_dialog, null)
        setView(rootView)
        uploadButton = rootView.findViewById(R.id.button_upload_uploadDialog)
        val retryButton = rootView.findViewById<Button>(R.id.button_retry_uploadFiles)
        val recyclerView = rootView.findViewById<RecyclerView>(R.id.rv_files_uploadDialog)
        switcher = rootView.findViewById(R.id.switcher_connectionStatus_uploadDialog)
        hintTV = rootView.findViewById(R.id.tv_hint_uploadFilesDialog)
        endPointDetailsTV = rootView.findViewById(R.id.tv_endPointDetails_uploadDialog)
        noFilesTV = rootView.findViewById(R.id.tv_noFilesToUpload_uploadDialog)
        activityCountTextView = rootView.findViewById(R.id.tv_activity_counts)
        recyclerView.adapter = adapter
        uploader.onItemChanged = this::onRecordingItemChanged
        uploader.onAllItemsFinishedWork = this::onRecordingsFinishedWorking
        uploader.reloadRecordings()

        if (recordings.size > 0) {
            noFilesTV.visibility = View.GONE
            uploadButton.visibility = View.VISIBLE
        }
        uploadButton.setOnClickListener {
            if (PreferencesHelper.getWebDAVToken(context) == "") {
                Toast.makeText(
                    context,
                    "Token for WebDAV not specified. Please specify in app settings",
                    Toast.LENGTH_LONG
                ).show()
            } else {
                uploadButton.isEnabled = false
                uploadButton.text = "Uploading"
                uploader.synchronize()
            }
        }
        retryButton.setOnClickListener {
            switcher.displayedChild = 0

            uploadButton.isEnabled = false
            uploadButton.text = "Uploading"
            uploader.synchronize()
        }
        val host = PreferencesHelper.getWebDAVUrl(context)
        val user = PreferencesHelper.getWebDAVUser(context)
        endPointDetailsTV.text = "User: $user\nHost: $host"
        adapter.notifyDataSetChanged()
        updateActivityCounts()
    }

    private fun onRecordingItemChanged(recording: RecordingUIItem) {
        adapter.notifyItemChanged(recordings.indexOf(recording))
    }

    private fun onRecordingsFinishedWorking() {
        uploadButton.isEnabled = true
        uploadButton.text = "Upload"
    }

    private fun updateActivityCounts() {
        val numberOfRecodings = recordingsManager.getNumberOfRecordingsPerActivity()
        var text = " "
        for ((activity, number) in numberOfRecodings) {
            text += "$activity: $number | "
        }
        activityCountTextView.text = text.trimEnd('|', ' ')
        activityCountTextView.visibility =
            if (numberOfRecodings.isEmpty()) View.GONE else View.VISIBLE
    }
}
