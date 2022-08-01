package sensors_in_paradise.sonar.use_cases

import android.content.Context
import android.util.Log
import java.io.File

class UseCaseHandler(
    val context: Context
) {
    private var useCasesStorage = UseCasesStorage(context)
    val useCasesBaseDir = getUseCasesBaseDir(context)
    private var defaultUseCase = UseCase(
        context,
        useCasesBaseDir,
        DEFAULT_USE_CASE_TITLE,
        this::onUseCaseDeleted,
        this::onUseCaseDuplicated
    )
    val availableUseCases = arrayListOf<UseCase>()
    private var currentUseCase: UseCase = defaultUseCase

    private var onUseCaseChanged: ((useCase: UseCase) -> Unit)? = null

    init {
        loadUseCases()
        setUseCase(useCasesStorage.getSelectedUseCase())
    }

    private fun loadUseCases() {
        val useCaseDirs = useCasesBaseDir.listFiles { dir, name ->
            dir.resolve(name).isDirectory
        }
        Log.d(
            "UseCaseHandler-loadUseCases",
            "useCaseDirs: ${useCasesBaseDir.listFiles()?.map { it.name }}"
        )
        if (useCaseDirs != null) {
            for (dir in useCaseDirs) {
                availableUseCases.add(
                    UseCase(
                        context,
                        useCasesBaseDir,
                        dir.name,
                        this::onUseCaseDeleted,
                        this::onUseCaseDuplicated
                    )
                )
            }
        }
    }

    fun setUseCase(title: String): Boolean {
        val useCase = availableUseCases.find { it.title == title }
        setUseCase(useCase ?: defaultUseCase)
        return useCase != null
    }

    private fun setUseCase(useCase: UseCase) {
        useCasesStorage.setSelectedUseCase(useCase.title)
        currentUseCase = useCase
        onUseCaseChanged?.invoke(useCase)
    }

    fun createUseCase(title: String, addToAvailableUseCases: Boolean = true): UseCase {
        if (hasUseCase(title)) {
            throw FileAlreadyExistsException(useCasesBaseDir.resolve(title), reason = "Use case $title already exists")
        }
        val useCase = UseCase(
            context,
            useCasesBaseDir,
            title,
            onUseCaseDeleted = this::onUseCaseDeleted,
            onUseCaseDuplicated = this::onUseCaseDuplicated
        )
        if (addToAvailableUseCases) {
            availableUseCases.add(useCase)
        }
        return useCase
    }

    fun getCurrentUseCase(): UseCase {
        return currentUseCase
    }

    fun hasUseCase(title: String): Boolean {
        return availableUseCases.find { it.title == title } != null
    }

    fun setOnUseCaseChanged(onUseCaseChanged: (useCase: UseCase) -> Unit) {
        this.onUseCaseChanged = onUseCaseChanged
    }

    private fun onUseCaseDeleted(useCase: UseCase) {
        availableUseCases.remove(useCase)
    }

    private fun onUseCaseDuplicated(originalUseCase: UseCase, duplicateTitle: String): UseCase {
        val duplicatedUseCase = createUseCase(duplicateTitle, false)
        duplicatedUseCase.setRecordingsSubDir(originalUseCase.getRecordingsSubDir().name)
        return duplicatedUseCase
    }

    companion object {
        fun getUseCasesBaseDir(context: Context): File {
            return File(context.getExternalFilesDir(null) ?: context.dataDir, "useCases")
        }

        const val DEFAULT_USE_CASE_TITLE = "default"
    }
}
