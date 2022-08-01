package sensors_in_paradise.sonar.uploader

import androidx.annotation.UiThread
import java.io.File

interface DavCloudClientInterface {
    @UiThread
    fun onDirCreated(dirPath: String, localReferenceDir: File?)
    @UiThread
    fun onDirCreationFailed(dirPath: String, localReferenceDir: File?, e: Exception)
    @UiThread
    fun onFileUploaded(localFile: File, filePath: String)
    @UiThread
    fun onFileUploadFailed(localFile: File, filePath: String, e: Exception)
    @UiThread
    fun onCredentialsNotAvailable(e: Exception)
}
