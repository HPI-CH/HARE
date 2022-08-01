package sensors_in_paradise.sonar.use_cases

import android.content.Context
import sensors_in_paradise.sonar.JSONStorage
import java.io.File

class UseCasesStorage(file: File) : JSONStorage(file) {
    constructor(context: Context) : this(getUseCaseStorageFile(context))

    override fun onFileNewlyCreated() {
        json.put(SELECTED_USE_CASE_KEY, UseCaseHandler.DEFAULT_USE_CASE_TITLE)
    }

    override fun onJSONInitialized() {
    }

    fun setSelectedUseCase(title: String) {
        json.put(SELECTED_USE_CASE_KEY, title)
        save()
    }

    fun getSelectedUseCase(): String {
        return json.optString(SELECTED_USE_CASE_KEY) ?: UseCaseHandler.DEFAULT_USE_CASE_TITLE
    }

    companion object {
        private const val SELECTED_USE_CASE_KEY = "selectedUseCase"
        fun getUseCaseStorageFile(context: Context): File {
            return context.dataDir.resolve("currentUseCase.json")
        }
    }
}
