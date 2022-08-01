package sensors_in_paradise.sonar.screen_data

import android.content.Context
import android.util.Log
import sensors_in_paradise.sonar.AsyncUI
import sensors_in_paradise.sonar.machine_learning.DataSet
import sensors_in_paradise.sonar.machine_learning.TFLiteModel
import sensors_in_paradise.sonar.screen_recording.Recording
import sensors_in_paradise.sonar.screen_recording.RecordingDataManager
import sensors_in_paradise.sonar.util.dialogs.ConfusionMatrixDialog
import sensors_in_paradise.sonar.util.dialogs.MessageDialog
import sensors_in_paradise.sonar.util.dialogs.ProgressDialog
import kotlin.math.roundToInt

class ModelPrediction(
    private val context: Context,
    private val recordingsManager: RecordingDataManager,
    val model: TFLiteModel
) : AsyncUI() {

    init {
        MessageDialog(
            context,
            "Do you want to include recordings that were used to train this model?",
            "Predict",
            "Include",
            { _, _ ->
                prepareAndExecutePrediction(recordingsManager)
            },
            "Exclude",
            { _, _ ->
                prepareAndExecutePrediction(
                    recordingsManager.filter { !it.metadataStorage.hasBeenUsedForOnDeviceTraining() })
            })
    }

    @Suppress("TooGenericExceptionCaught")
    private fun prepareAndExecutePrediction(recordings: List<Recording>) {
        val progressDialog = ProgressDialog(context)
        ui {
            progressDialog.show()
            progressDialog.setProgress(0, "Prediction process")
        }

        async {
            try {
                val dataFiles = RecordingDataManager.convertRecordings(
                    recordings,
                    regenerateExistingFiles = false
                ) {
                    ui {
                        progressDialog.setSubProgress(
                            it,
                            "Loading recording data files ${((it / 100f) * recordings.size).toInt()}/${recordings.size}"
                        )
                    }
                }
                ui { progressDialog.setProgress(30) }

                val dataSet = DataSet().apply {
                    addAll(dataFiles)
                }

                ui {
                    progressDialog.setSubProgress(
                        0,
                        "Converting recordings into batches"
                    )
                }
                val batches = dataSet.convertToBatches(7, model.windowSize, filterForActivities = model.getLabels()) {
                    ui {
                        progressDialog.setSubProgress(it)
                    }
                }
                ui {
                    progressDialog.setProgress(80)
                    progressDialog.setSubProgress(0, "Evaluating model")
                }

                val (accuracyBefore, cm) = model.evaluate(batches) { batchProgress, _ ->
                    ui { progressDialog.setSubProgress(batchProgress) }
                }

                ui { progressDialog.setProgress(100) }

                cm.title =
                    "Confusion Matrix (acc ${(accuracyBefore * 100).roundToInt()}%)"
                ui {
                    progressDialog.setProgress(100)
                    progressDialog.dismiss()
                    ConfusionMatrixDialog(
                        context,
                        listOf(cm)
                    )
                }
                Log.d("ModelPrediction", "Confusion Matrix: \n$cm")
                Log.d("ModelPrediction", "Evaluated on ${batches.size} batches of size 7")
                Log.d("ModelPrediction", "Dataset was of size ${dataSet.size}")
            } catch (e: Exception) {
                e.printStackTrace()
                ui {
                    progressDialog.dismiss()
                    MessageDialog(
                        context,
                        "Prediction failed with exception: \n${e.message}",
                        "Prediction failed"
                    )
                }
            }
        }
    }
}
