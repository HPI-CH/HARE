package sensors_in_paradise.sonar.screen_data

import android.content.Context
import sensors_in_paradise.sonar.AsyncUI
import sensors_in_paradise.sonar.machine_learning.DataSet
import sensors_in_paradise.sonar.machine_learning.TFLiteModel
import sensors_in_paradise.sonar.screen_recording.Recording
import sensors_in_paradise.sonar.screen_recording.RecordingDataManager
import sensors_in_paradise.sonar.util.dialogs.ConfusionMatrixDialog
import sensors_in_paradise.sonar.util.dialogs.MessageDialog
import sensors_in_paradise.sonar.util.dialogs.ProgressDialog
import sensors_in_paradise.sonar.util.dialogs.SingleChoiceDialog
import kotlin.math.roundToInt

class ModelTraining(
    private val context: Context,
    private val recordingsManager: RecordingDataManager,
    val model: TFLiteModel,
    private val onTrainedModelSaveRequested: (recordingsUsedForTraining: List<Recording>) -> Unit
) : AsyncUI() {
    private val progressDialog = ProgressDialog(context)

    init {
        val peopleDurations = recordingsManager.getPeopleDurations(onlyUntrainedRecordings = true)
        SingleChoiceDialog(
            context,
            "Subjects to train on",
            peopleDurations.keys.toTypedArray(),
            onItemChosen = {
                val recordings = recordingsManager.getRecordingsBySubject(it, true)
                if (recordings.size >= 2) {
                    prepareAndExecuteTraining(recordings, it)
                } else {
                    MessageDialog(
                        context,
                        "There are only ${recordings.size} recordings of" +
                                " subject $it. To allow for validation after training, you need at least " +
                                "2 recordings so that one can be used for validation.",
                        "Not enough recordings"
                    )
                }
            })
    }

    @Suppress("TooGenericExceptionCaught")
    private fun prepareAndExecuteTraining(recordings: List<Recording>, subject: String) {
        progressDialog.show()

        progressDialog.setProgress(0, "Training process")
        progressDialog.setSubProgress(0, "Merging recording data files")
        async {
            try {

                val dataFiles = RecordingDataManager.convertRecordings(recordings) {
                    ui { progressDialog.setSubProgress(it) }
                }
                ui { progressDialog.setProgress(10) }

                val dataSet = DataSet().apply {
                    addAll(dataFiles)
                }

                val (trainDataSet, validationDataSet) = dataSet.splitByPercentage(
                    DESIRED_VALIDATION_SPLIT
                )

                ui {
                    progressDialog.setSubProgress(
                        0,
                        "Converting validation recordings into batches"
                    )
                }
                val validationBatches =
                    validationDataSet.convertToBatches(
                        VALIDATION_BATCH_SIZE,
                        model.windowSize,
                        filterForActivities = model.getLabels()
                    ) {
                        ui {
                            progressDialog.setSubProgress(it)
                        }
                    }
                ui {
                    progressDialog.setProgress(20)
                    progressDialog.setSubProgress(0, "Evaluating model before training")
                }

                val (accuracyBefore, cmBefore) = model.evaluate(validationBatches) { batchProgress, _ ->
                    ui { progressDialog.setSubProgress(batchProgress) }
                }

                ui {
                    progressDialog.setSubProgress(0, "Converting training recordings into batches")
                }
                val trainBatches = trainDataSet.convertToBatches(
                    BATCH_SIZE,
                    model.windowSize,
                    filterForActivities = model.getLabels()
                ) {
                    ui {
                        progressDialog.setSubProgress(it)
                    }
                }
                ui { progressDialog.setProgress(30) }
                model.train(trainBatches, NUM_EPOCHS) { epoch, batch, window ->
                    ui {
                        progressDialog.setSubProgress(
                            epoch,
                            "Training model\nEpoch: $epoch%\nBatch: $batch%\nWindow: $window%"
                        )
                    }
                }
                ui {
                    progressDialog.setProgress(90)
                    progressDialog.setSubProgress(0, "Evaluating model after training")
                }
                val (accuracyAfter, cmAfter) = model.evaluate(validationBatches) { batchProgress, _ ->
                    ui {
                        progressDialog.setSubProgress(batchProgress)
                    }
                }
                val evaluationDescription =
                    "\n\nEvaluated on ${validationDataSet.size} recordings of $subject comprised of ${VALIDATION_BATCH_SIZE * validationBatches.size} windows.\n"
                cmBefore.title =
                    "Confusion Matrix before training"
                cmBefore.description =
                    "Accuracy: ${(accuracyBefore * 100).roundToInt()}%$evaluationDescription"
                cmAfter.title =
                    "Confusion Matrix after training"
                cmAfter.description =
                    "Accuracy: ${(accuracyAfter * 100).roundToInt()}%$evaluationDescription" +
                            "Trained on ${trainDataSet.size} recordings comprised of " +
                            "${trainBatches.size} batches of size $BATCH_SIZE for $NUM_EPOCHS epochs"
                ui {
                    progressDialog.dismiss()
                    ConfusionMatrixDialog(
                        context,
                        listOf(
                            cmBefore,
                            cmAfter
                        ),
                        positiveButtonText = "Save Model",
                        onPositiveButtonClickListener = { _, _ ->
                            onTrainedModelSaveRequested(trainDataSet as List<Recording>)
                        })
                }
            } catch (e: Exception) {
                ui {
                    progressDialog.dismiss()
                    MessageDialog(
                        context,
                        "Training failed with exception: \n${e.message}",
                        "Training failed"
                    )
                }
                e.printStackTrace()
            }
        }
    }

    companion object {
        const val NUM_EPOCHS = 1
        const val BATCH_SIZE = 7
        const val VALIDATION_BATCH_SIZE = 7
        const val DESIRED_VALIDATION_SPLIT = 0.2f
    }
}
