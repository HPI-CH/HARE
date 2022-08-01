package sensors_in_paradise.sonar.use_cases

import android.app.AlertDialog
import android.content.Context
import android.content.DialogInterface
import android.util.Log
import android.view.LayoutInflater
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import androidx.core.widget.addTextChangedListener
import androidx.recyclerview.widget.RecyclerView
import sensors_in_paradise.sonar.R

class UseCaseDialog(context: Context, useCaseHandler: UseCaseHandler) {
    private var selectedUseCase: UseCase? = useCaseHandler.getCurrentUseCase()
    private var positiveButton: Button? = null
    init {
        val availableUseCases = useCaseHandler.availableUseCases

        val builder = AlertDialog.Builder(context)
        builder.setTitle("Choose a use case")
        builder.setMessage("Each use case stores its own tflite model, labels, people and recordings.")
        val root = LayoutInflater.from(context).inflate(R.layout.usecase_dialog, null)
        val recyclerView = root.findViewById<RecyclerView>(R.id.recyclerView_useCases_useCaseDialog)
        val useCasesAdapter = UseCasesAdapter(context, availableUseCases, useCaseHandler.getCurrentUseCase()) {
            selectedUseCase = it
            positiveButton?.isEnabled = selectedUseCase != null
        }
        val addUseCaseButton = root.findViewById<ImageButton>(R.id.imageButton_addUseCase_useCaseDialog)
        val addUseCaseEditText = root.findViewById<EditText>(R.id.editText_addNewUseCase_useCaseDialog)

        addUseCaseEditText.apply {
            addTextChangedListener {
                val value = text.toString()
                val len = value.length
                addUseCaseButton.apply {
                    isEnabled = availableUseCases.find { it.title == value } == null && len> 0
                }
        }
        }

        addUseCaseButton.apply {
            isEnabled = false
            setOnClickListener {
                val useCase = useCaseHandler.createUseCase(addUseCaseEditText.text.toString())
                val index = availableUseCases.indexOf(useCase)
                Log.d("UseCaseDialog", "setting selectedIndex of usecases to $index")
                useCasesAdapter.selectItem(index)
                addUseCaseEditText.setText("")
            }
        }

        recyclerView.adapter = useCasesAdapter
        builder.setView(root)

        builder.setPositiveButton("OK") { _, _ ->
                useCaseHandler.setUseCase(selectedUseCase!!.title)
        }
        builder.setNegativeButton("Cancel", null)
        val dialog = builder.create()
        dialog.setOnShowListener {
            positiveButton = dialog.getButton(DialogInterface.BUTTON_POSITIVE)
            dialog.window
                ?.clearFlags(
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_ALT_FOCUSABLE_IM
                )
        }
        dialog.show()
    }
}
