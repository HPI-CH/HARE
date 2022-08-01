package sensors_in_paradise.sonar.uploader

import android.app.Activity
import android.content.Context
import android.util.Log
import com.google.common.net.MediaType
import sensors_in_paradise.sonar.util.dialogs.MessageDialog
import sensors_in_paradise.sonar.screen_recording.RecordingDataManager
import sensors_in_paradise.sonar.util.PreferencesHelper
import java.io.File
import java.net.URLConnection

class DavCloudRecordingsUploader(activity: Activity, val recordingsManager: RecordingDataManager) :
    DavCloudClientInterface {
    val context: Context = activity
    var onItemChanged: ((recording: RecordingUIItem) -> Unit)? = null
    var onAllItemsFinishedWork: (() -> Unit)? = null
    private val davCloudMetadata =
        LocalDavCloudMetadataStorage(activity, recordingsManager.recordingsDir)
    private val davCloud = DavCloudClient(activity, this)
    val recordingUiItems = RecordingUIItemArrayList()
    private val dirCreationRequests = mutableSetOf<File>()
    private val videoUpload = PreferencesHelper.shouldUploadCameraRecordings(context)

    init {
        reloadRecordings()
    }

    fun reloadRecordings() {
        davCloudMetadata.updateWebDavEndpoint()
        recordingUiItems.clear()
        for (recording in recordingsManager) {
            val recording = RecordingUIItem(recording, davCloudMetadata.localUploadedFilesBaseDir)

            for (file in recording.filesAndDirsToBeUploaded) {
                val isUploaded =
                    if (file.isFile) davCloudMetadata.isFileUploaded(file) else davCloudMetadata.isDirCreated(
                        file
                    )
                recording.setStatusOfFileOrDir(
                    file,
                    if (isUploaded) UploadStatus.UPLOADED else UploadStatus.NOT_UPLOADED
                )
            }
            if (recording.isUploaded()) {
                recordingUiItems.add(recording)
            } else {
                recordingUiItems.add(0, recording)
            }
        }
    }

    fun synchronize() {
        Log.d("DAVCLOUD", "Synchronize")
        for (recordingUiItem in recordingUiItems) {
            Log.d("DAVCLOUD", "trying recording")
            if (recordingUiItem.isValid) {
                uploadRecording(recordingUiItem)
            }
        }
        if (onAllItemsFinishedWork != null) {
            if (areAllRecordingsFinished()) {
                onAllItemsFinishedWork?.let { it() }
            }
        }
    }

    private fun uploadRecording(recording: RecordingUIItem) {
        Log.d("DAVCLOUD", "Uploading recording!")
        if (davCloudMetadata.isDirCreated(recording.dir)) {
            uploadFilesOfRecording(recording)
        } else {
            for (dir in recording.dirsToBeCreated) {
                val isCreated = davCloudMetadata.isDirCreated(dir)
                val isCreationAlreadyRequested = dirCreationRequests.contains(dir)
                if (!isCreated && !isCreationAlreadyRequested) {
                    dirCreationRequests.add(dir)
                    recording.setStatusOfFileOrDir(dir, UploadStatus.UPLOADING)
                    davCloud.createDir(davCloudMetadata.getRelativePath(dir), dir)
                    break
                }
            }
        }
        onItemChanged?.let { it(recording) }
    }

    private fun uploadFilesOfRecording(recording: RecordingUIItem) {
        for (file in recording.filesToBeUploaded) {
            val fileType = URLConnection.guessContentTypeFromName(file.toString())
            if (fileType != null && fileType.startsWith("video") && !videoUpload) continue
            if (!davCloudMetadata.isFileUploaded(file)) {
                Log.d("DAVCLOUD", "Uploading file: ${file.name}")
                recording.setStatusOfFileOrDir(file, UploadStatus.UPLOADING)
                davCloud.uploadFile(
                    file,
                    davCloudMetadata.getRelativePath(file),
                    MediaType.CSV_UTF_8
                )
            } else {
                recording.setStatusOfFileOrDir(file, UploadStatus.UPLOADED)
            }
            onItemChanged?.let { it(recording) }
        }
    }

    override fun onDirCreated(dirPath: String, localReferenceDir: File?) {
        Log.d("DAVCLOUD", "Dir created: $dirPath")
        davCloudMetadata.setDirCreated(localReferenceDir!!)
        val recordings = recordingUiItems.getRecordingsInDir(localReferenceDir)
        for (recording in recordings) {
            recording.setStatusOfFileOrDir(localReferenceDir, UploadStatus.UPLOADED)
            uploadRecording(recording)
            onRecordingStatusChanged(recording)
        }
    }

    override fun onDirCreationFailed(dirPath: String, localReferenceDir: File?, e: Exception) {
        Log.e("DAVCLOUD", "Dir creation failed: ${e.message}")
        val recordings = recordingUiItems.getRecordingsInDir(localReferenceDir!!)
        for (recording in recordings) {
            if (recording.getStatusOfFileOrDir(localReferenceDir) != UploadStatus.UPLOADED) {
                recording.error = e
                recording.setStatusOfFileOrDir(localReferenceDir, UploadStatus.FAILED)
                onRecordingStatusChanged(recording)
            }
        }
    }

    override fun onFileUploaded(localFile: File, remoteFilePath: String) {
        Log.d("DAVCLOUD", "File Uploaded: $remoteFilePath")
        davCloudMetadata.setFileUploaded(localFile)
        val recording = recordingUiItems.getByRecordingDir(localFile.parentFile!!)!!
        recording.setStatusOfFileOrDir(localFile, UploadStatus.UPLOADED)
        onRecordingStatusChanged(recording)
    }

    override fun onFileUploadFailed(localFile: File, filePath: String, e: Exception) {
        Log.e("DAVCLOUD", "Dir creation failed: ${e.message}")

        val recording = recordingUiItems.getByRecordingDir(localFile.parentFile!!)!!
        recording.error = e
        recording.setStatusOfFileOrDir(localFile, UploadStatus.FAILED)
        onRecordingStatusChanged(recording)
    }

    override fun onCredentialsNotAvailable(e: java.lang.Exception) {
        MessageDialog(context, e.message ?: "Credentials are not available")
    }

    private fun onRecordingStatusChanged(recording: RecordingUIItem) {
        onItemChanged?.let { it(recording) }
        if (onAllItemsFinishedWork != null) {
            if (areAllRecordingsFinished()) {
                onAllItemsFinishedWork?.let { it() }
            }
        }
    }

    private fun areAllRecordingsFinished(): Boolean {
       return recordingUiItems.areAllUploadedOrFailed()
    }
}
