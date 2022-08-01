package sensors_in_paradise.sonar.uploader

import java.io.File
import java.lang.Exception

interface FileUploaderInterface {
    fun onConnectionFailed(hint: String)
    fun onFileUploaded(file: File)
    fun onFileUploadFailed(file: File, exception: Exception)
    fun onFileUploadStarted(file: File)
}
