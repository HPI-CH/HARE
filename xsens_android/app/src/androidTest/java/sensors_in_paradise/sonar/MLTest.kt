package sensors_in_paradise.sonar

import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Before
import org.junit.Test
import sensors_in_paradise.sonar.machine_learning.DataSet
import sensors_in_paradise.sonar.machine_learning.TFLiteModel
import sensors_in_paradise.sonar.screen_recording.RecordingDataFile
import sensors_in_paradise.sonar.use_cases.UseCase
import java.io.File

class MLTest {
    private val assetContext = InstrumentationRegistry.getInstrumentation().context
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val recordingFile = File(appContext.cacheDir, "recordingData.csv")
    private val modelFile = File(appContext.cacheDir, "model_test.tflite")

    @Before
    fun init() {
        UseCase.extractFileFromAssets(assetContext, "0_orhan_1652085453257.csv", recordingFile)
        UseCase.extractFileFromAssets(assetContext, "resnet_model.tflite", modelFile)
    }

    @Test
    fun windowizeTest() {
        val features =
            arrayOf("Quat_Z_LF", "dq_W_LF", "dv[1]_LF").map { it.uppercase() }.toTypedArray()
        val windowSize = 90
        val data = RecordingDataFile(recordingFile)
        val startIndexes = data.getWindowStartIndexes(windowSize)

        assert(startIndexes.size > 0)
        var i = 1
        for (startIndex in startIndexes) {
            Log.d(
                "RecordingDataFileTest-windowizeTest",
                "Working on window $i of ${startIndexes.size}"
            )
            val (window, activity) = data.getWindowAtIndex(startIndex, windowSize, features)
            assert(window.size == features.size)

            // Test if compiling the window into a float buffer runs through
            window.compileWindow()
            i++
        }
    }

    @Test
    fun predictionPipelineTest() {
        val model = TFLiteModel(modelFile)

        val features = model.getFeaturesToPredict().map { it.uppercase() }.toTypedArray()

        val data = RecordingDataFile(recordingFile)
        val startIndexes = data.getWindowStartIndexes(model.windowSize)
        var correctPredictions = 0
        assert(startIndexes.size > 0)

        var i = 1
        for (startIndex in startIndexes) {
            val (window, activity) = data.getWindowAtIndex(startIndex, model.windowSize, features)
            assert(window.size == features.size)

            // Test if compiling the window into a float buffer runs through
            val input = window.compileWindowToArray()
            val prediction = model.infer(arrayOf(input))
            val predictedLabel = model.convertPredictionToLabel(prediction[0])
            if (predictedLabel == activity) {
                correctPredictions++
            }
            Log.d(
                "RecordingDataFileTest-predictionPipelineTest",
                "Working on window $i of ${startIndexes.size}"
            )
            i++
        }
        val accuracy = (correctPredictions * 100 / startIndexes.size)
        Log.d("RecordingDataFileTest", "Prediction accuracy on the example recording: $accuracy%")
    }

    @Test
    fun trainingPipelineTest() {
        val model = TFLiteModel(modelFile)
        val data = RecordingDataFile(recordingFile)
        val dataSet = DataSet().apply { add(data) }

        val batches = dataSet.convertToBatches(7, model.windowSize, progressCallback = { progress ->
            Log.d(
                "RecordingDataFileTest-trainingPipelineTest",
                "Batching dataset: $progress%"
            )
        })

        val accuracyBeforeTraining = model.evaluate(batches) { batch, window ->
            Log.d(
                "RecordingDataFileTest",
                "Evaluating model before training. Batch: $batch%, Window Progress: $window%"
            )
        }
        val losses = model.train(batches, 5) { epoch, batch, window ->
            Log.d(
                "RecordingDataFileTest",
                "Training model. Epoch $epoch%, batch $batch%, window $window%"
            )
        }
        Log.d(
            "RecordingDataFileTest",
            "Losses during training epochs: ${losses.joinToString()}"
        )
        val accuracyAfterTraining = model.evaluate(batches) { batch, window ->
            Log.d(
                "RecordingDataFileTest",
                "Evaluating model after training. Batch: $batch%, Window Progress: $window%"
            )
        }
        Log.d(
            "RecordingDataFileTest",
            "Prediction accuracy on the example recording before training: " +
                    "$accuracyBeforeTraining and after training: $accuracyAfterTraining%"
        )
    }
}
