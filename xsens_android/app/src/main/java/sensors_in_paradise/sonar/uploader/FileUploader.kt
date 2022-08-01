package sensors_in_paradise.sonar.uploader

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import com.android.volley.*
import com.android.volley.toolbox.StringRequest
import com.android.volley.toolbox.Volley
import java.io.File
import com.android.volley.toolbox.HttpHeaderParser

import java.io.IOException
import java.nio.file.Files
import kotlin.collections.ArrayList

class FileUploader(private val rootDir: File, private val callback: FileUploaderInterface?) {
    private var queue: RequestQueue? = null
    private val urlPrefix = "http://10.3.141.1:5000/files/"

    fun uploadFiles(context: Context) {
        if (hasWifiConnection(context)) {
            if (queue == null) {
                queue = Volley.newRequestQueue(context)
            }
            val files = getFilesToBeUploaded(rootDir)
            for (file in files) {
                callback?.onFileUploadStarted(file)
                try {
                    uploadFile(file, {
                        file.delete()
                        callback?.onFileUploaded(file)
                    }, { error -> callback?.onFileUploadFailed(file, error) })
                } catch (e: IOException) {
                    callback?.onFileUploadFailed(file, e)
                }
            }
        } else {
            callback?.onConnectionFailed("Device not connected to Wifi")
        }
    }

    @Throws(IOException::class)
    private fun readFile(file: File): ByteArray {
        return Files.readAllBytes(file.toPath())
    }

    @Throws(IOException::class)
    private fun uploadFile(file: File, onSuccess: Response.Listener<String>, onError: Response.ErrorListener) {
        val url = urlPrefix + getURLSuffixForFile(file)
        val requestBody = readFile(file)

        val stringRequest: StringRequest = object : StringRequest(Method.POST, url, onSuccess, onError) {
            override fun getBodyContentType(): String {
                return "application/json; charset=utf-8"
            }

            @Throws(AuthFailureError::class)
            override fun getBody(): ByteArray? {
                return requestBody
            }

            override fun parseNetworkResponse(response: NetworkResponse): Response<String> {
                val responseString = response.statusCode.toString()
                return Response.success(responseString, HttpHeaderParser.parseCacheHeaders(response))
            }
        }
        queue?.add(stringRequest)

        if (queue == null) {
            onError.onErrorResponse(VolleyError("RequestQueue not initialized"))
        }
    }

    private fun hasWifiConnection(context: Context): Boolean {
        val connManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager?
        val activeNetwork = connManager!!.activeNetwork
        with(connManager) {
            val capabilities = getNetworkCapabilities(activeNetwork)
            if (capabilities != null) {
                return capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)
            }
        }
        return false
    }

    fun getURLSuffixForFile(file: File): String {
        var relativeFile = file.relativeTo(rootDir)
        var result = relativeFile.name
        while (relativeFile.parentFile != null) {
            relativeFile = relativeFile.parentFile
            result = relativeFile.name + "/" + result
        }
        return result
    }

    fun getFilesToBeUploaded(baseDir: File): ArrayList<File> {
        if (baseDir.isFile) {
            return arrayListOf(baseDir)
        }
        val result = ArrayList<File>()
        val children = baseDir.listFiles()
        if (children != null) {
            for (fileOrDir in children) {
                result.addAll(getFilesToBeUploaded(fileOrDir))
            }
        }
        return result
    }
}
