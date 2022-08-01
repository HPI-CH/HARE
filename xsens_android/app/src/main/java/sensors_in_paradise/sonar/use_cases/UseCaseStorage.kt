package sensors_in_paradise.sonar.use_cases

import android.util.Log
import sensors_in_paradise.sonar.JSONStorage
import java.io.File

class UseCaseStorage(file: File) : JSONStorage(file) {

    override fun onFileNewlyCreated() {
        json.put(SELECTED_SUB_DIR_KEY, UseCase.DEFAULT_RECORDINGS_SUB_DIR_NAME)
    }

    override fun onJSONInitialized() {}

    fun setSelectedSubDir(subDir: String) {
        /*
            json: {
                    useCase:{
                        selectedSubDir: "default",
                    }
                }
            }
        */
        Log.d("useCase", json.toString())
        json.put(SELECTED_SUB_DIR_KEY, subDir)
        save()
    }

    fun getSelectedSubDir(): String {
        Log.d("useCase", json.toString())

        return json.optString(SELECTED_SUB_DIR_KEY)
                ?: UseCase.DEFAULT_RECORDINGS_SUB_DIR_NAME
    }

    companion object {
        private const val SELECTED_SUB_DIR_KEY = "selectedSubDir"
    }
}
