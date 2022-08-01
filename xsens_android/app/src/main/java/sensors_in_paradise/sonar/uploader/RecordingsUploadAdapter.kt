package sensors_in_paradise.sonar.uploader

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.ViewAnimator
import androidx.recyclerview.widget.RecyclerView
import sensors_in_paradise.sonar.R

class RecordingsUploadAdapter(
    private val recordingUiItems: RecordingUIItemArrayList,
) :
    RecyclerView.Adapter<RecordingsUploadAdapter.ViewHolder>() {

    /**
     * Provide a reference to the type of views that you are using
     * (custom ViewHolder).
     */
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val labelTV: TextView = view.findViewById(R.id.tv_label_recordingUpload)
        val statusTV: TextView = view.findViewById(R.id.tv_status_recordingUpload)
        val animator: ViewAnimator = view.findViewById(R.id.switcher_uploadFile)
    }

    // Create new views (invoked by the layout manager)
    override fun onCreateViewHolder(viewGroup: ViewGroup, viewType: Int): ViewHolder {
        // Create a new view, which defines the UI of the list item
        val view = LayoutInflater.from(viewGroup.context)
            .inflate(R.layout.recording_upload, viewGroup, false)

        return ViewHolder(view)
    }

    // Replace the contents of a view (invoked by the layout manager)
    override fun onBindViewHolder(viewHolder: ViewHolder, position: Int) {

        // Get element from your dataset at this position and replace the
        // contents of the view with that element
        val recording = recordingUiItems[position]
        viewHolder.labelTV.text = recording.label

        val status = recording.getStatusLabel()
        viewHolder.statusTV.text = status
        viewHolder.animator.displayedChild = recording.getSummarizedStatus().ordinal
    }

    // Return the size of your dataset (invoked by the layout manager)
    override fun getItemCount(): Int {
        return recordingUiItems.size
    }
}
