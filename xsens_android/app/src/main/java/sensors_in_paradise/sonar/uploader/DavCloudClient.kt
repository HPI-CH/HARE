package sensors_in_paradise.sonar.uploader

import android.app.Activity
import android.content.Context
import com.google.common.net.MediaType
import com.thegrizzlylabs.sardineandroid.Sardine
import com.thegrizzlylabs.sardineandroid.impl.OkHttpSardine
import sensors_in_paradise.sonar.util.PreferencesHelper
import java.io.File
import java.io.IOException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class DavCloudClient(val activity: Activity, val callback: DavCloudClientInterface) {
    val context: Context = activity
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val sardine: Sardine = OkHttpSardine()
    private val baseUrl = PreferencesHelper.getWebDAVUrl(context)
    private val user = PreferencesHelper.getWebDAVUser(context)
    private val token = PreferencesHelper.getWebDAVToken(context)
    private val credentialsAvailable = token.isNotEmpty() && baseUrl.isNotEmpty() && user.isNotEmpty()
    private val credentialsNotAvailableException =
        Exception("WebDAV cloud credentials (token, url or username) are not available. " +
                "Please visit the app settings and make sure these are not empty" +
                "(especially the token must be filled in once).")

    init {
        if (credentialsAvailable) {
            sardine.setCredentials(user, token)
        } else {
            callback.onCredentialsNotAvailable(credentialsNotAvailableException)
        }
    }

    fun createDir(remoteDirPath: String, localReferenceDir: File? = null) {
        if (credentialsAvailable) {
            executor.execute {
                try {
                    sardine.createDirectory("${baseUrl}${normalizePath(remoteDirPath)}")
                    activity.runOnUiThread {
                        callback.onDirCreated(
                            remoteDirPath,
                            localReferenceDir
                        )
                    }
                } catch (e: IOException) {
                    activity.runOnUiThread {
                        callback.onDirCreationFailed(
                            remoteDirPath,
                            localReferenceDir,
                            e
                        )
                    }
                }
            }
        } else {
            callback.onDirCreationFailed(
                remoteDirPath,
                localReferenceDir,
                credentialsNotAvailableException
            )
        }
    }

    fun uploadFile(localFile: File, remoteFilePath: String, mediaType: MediaType) {
        if (credentialsAvailable) {
            executor.execute {
                try {
                    sardine.put(
                        "${baseUrl}${normalizePath(remoteFilePath)}",
                        localFile,
                        mediaType.type()
                    )
                    activity.runOnUiThread { callback.onFileUploaded(localFile, remoteFilePath) }
                } catch (e: IOException) {
                    activity.runOnUiThread {
                        callback.onFileUploadFailed(
                            localFile,
                            remoteFilePath,
                            e
                        )
                    }
                }
            }
        } else {
            callback.onFileUploadFailed(localFile, remoteFilePath, credentialsNotAvailableException)
        }
    }

    companion object {
        fun normalizePath(path: String): String {
            if (path.startsWith("/")) {
                return path.replaceFirst("/", "")
            }
            if (path.startsWith("./")) {
                return path.replaceFirst("./", "")
            }
            return path
        }
    }
}
