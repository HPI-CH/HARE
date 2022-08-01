package sensors_in_paradise.sonar.screen_prediction

import android.app.Activity
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.*
import androidx.constraintlayout.motion.widget.MotionLayout
import androidx.recyclerview.widget.RecyclerView
import com.github.mikephil.charting.charts.BarChart
import com.google.android.material.button.MaterialButton
import com.xsens.dot.android.sdk.events.XsensDotData
import com.xsens.dot.android.sdk.models.XsensDotPayload
import sensors_in_paradise.sonar.*
import sensors_in_paradise.sonar.machine_learning.InMemoryWindow
import sensors_in_paradise.sonar.machine_learning.TFLiteModel
import sensors_in_paradise.sonar.screen_connection.ConnectionInterface
import sensors_in_paradise.sonar.machine_learning.TFLiteModel.InvalidModelMetadata
import sensors_in_paradise.sonar.screen_prediction.barChart.PredictionBarChart
import sensors_in_paradise.sonar.screen_train.PredictionHistoryStorage
import sensors_in_paradise.sonar.screen_train.PredictionHistoryStorage.Prediction
import sensors_in_paradise.sonar.use_cases.UseCase
import sensors_in_paradise.sonar.util.PreferencesHelper
import sensors_in_paradise.sonar.util.UIHelper
import sensors_in_paradise.sonar.util.dialogs.MessageDialog
import java.io.File
import java.lang.IllegalStateException
import kotlin.math.round

class PredictionScreen(
    private var currentUseCase: UseCase,
    private val devices: XSENSArrayList,
    private val sensorOccupationInterface: SensorOccupationInterface?
) : ScreenInterface, ConnectionInterface {
    private lateinit var activity: Activity
    private lateinit var context: Context
    private lateinit var recyclerView: RecyclerView
    private lateinit var predictionHistoryAdapter: PredictionHistoryAdapter
    private lateinit var predictionButton: MaterialButton
    private lateinit var progressBar: ProgressBar
    private lateinit var timer: Chronometer
    private lateinit var textView: TextView
    private var predictionBarChart: PredictionBarChart? = null

    private lateinit var toggleMotionLayout: MotionLayout
    private lateinit var metadataStorage: XSensDotMetadataStorage
    private var predictionHistoryStorage: PredictionHistoryStorage? = null

    private var lastPredictionTime = 0L
    private var numConnectedDevices = 0
    private var isRunning = false
    private var window: InMemoryWindow? = null
    private lateinit var mainHandler: Handler
    private var predictionInterval: Long? = null
    private val updatePredictionTask = object : Runnable {
        override fun run() {
            if (isRunning) {
                predict()
                mainHandler.postDelayed(this, predictionInterval!!)
            }
        }
    }

    private val updateProgressBarTask = object : Runnable {
        override fun run() {
            var progress = 0
            if (isRunning) {
                progress =
                    ((100 * (System.currentTimeMillis() - lastPredictionTime)) / predictionInterval!!).toInt()
                mainHandler.postDelayed(this, 40)
            }
            progressBar.progress = progress
        }
    }
    private var model: TFLiteModel? = null

    private fun togglePrediction() {
        if (isRunning) {
            stopDataCollection()
        } else {
            startDataCollection()
        }
    }

    private fun checkPredictionPreconditions(): Boolean {
        val isInvalidTagConnected =
            devices.getConnectedWithOfflineMetadata().any { !it.isTagValid() }
        if (isInvalidTagConnected) {
            MessageDialog(
                context,
                context.getString(R.string.sensor_tag_prefix_pattern_explanation),
                "Tags of connected sensors invalid"
            )
            return false
        }
        val requiredButNotConnectedDevices = getRequiredButNotConnectedDevices()
        if (requiredButNotConnectedDevices.isNotEmpty()) {
            UIHelper.showAlert(
                context = context,
                message = "The model for this use case requires sensors with the following " +
                        "tag prefixes to be connected:\n" +
                        "    ${model!!.getDeviceTags().joinToString("\n    ")}\n" +
                        "These are currently not connected:\n" +
                        "    ${requiredButNotConnectedDevices.joinToString("\n    ")}",
                title = "Not enough sensors connected"
            )
            return false
        }
        return true
    }

    private fun startDataCollection() {
        if (!checkPredictionPreconditions()) {
            return
        }
        if (model == null) {
            throw IllegalStateException("The TFLiteModel instance can't be null when starting data collection")
        }
        sensorOccupationInterface?.onSensorOccupationStatusChanged(true)
        lastPredictionTime = 0L
        window = model?.let {
            InMemoryWindow(
                windowSize = it.windowSize,
                featuresWithSensorTagPrefix = it.getFeaturesToPredict()
            )
        }
        for (device in devices.getConnected()) {
            device.measurementMode =
                XsensDotPayload.PAYLOAD_TYPE_CUSTOM_MODE_4
            device.startMeasuring()
        }
        timer.base = SystemClock.elapsedRealtime()
        timer.start()
        textView.visibility = View.VISIBLE
        textView.text = ""
        val barChart: BarChart = activity.findViewById(R.id.barChart_predict_predictions)
        predictionBarChart =
            PredictionBarChart(context, barChart, (model!!.getLabelsMap().size))
        predictionBarChart?.resetData()

        predictionHistoryStorage =
            PredictionHistoryStorage(
                currentUseCase,
                System.currentTimeMillis(),
                PreferencesHelper.shouldStorePrediction(context)
            )
        predictionHistoryAdapter.predictionHistory = arrayListOf()
        predictionHistoryAdapter.addPrediction(Prediction("", 0f), 0)
        isRunning = true
        mainHandler.postDelayed(updatePredictionTask, predictionInterval!!)
        mainHandler.postDelayed(updateProgressBarTask, 100)
        progressBar.visibility = View.VISIBLE
        predictionButton.setIconResource(R.drawable.ic_baseline_stop_24)

        toggleMotionLayout.transitionToEnd()
    }

    private fun getRequiredButNotConnectedDevices(): List<String> {
        val connectedDeviceTagPrefixes =
            devices.getConnectedWithOfflineMetadata().map { it.getTagPrefix() }
        val requiredDeviceTagPrefixes = model!!.getDeviceTags()
        return requiredDeviceTagPrefixes.filter { !connectedDeviceTagPrefixes.contains(it) }
    }

    private fun stopDataCollection() {
        sensorOccupationInterface?.onSensorOccupationStatusChanged(false)
        timer.stop()
        for (device in devices.getConnected()) {
            device.stopMeasuring()
        }
        textView.text = ""
        textView.visibility = View.GONE
        isRunning = false
        mainHandler.removeCallbacks(updatePredictionTask)
        mainHandler.removeCallbacks(updateProgressBarTask)
        progressBar.visibility = View.INVISIBLE
        predictionButton.setIconResource(R.drawable.ic_baseline_play_arrow_24)
        model?.close()

        toggleMotionLayout.transitionToStart()
    }

    private fun updatePrediction(output: FloatArray) {
        val predictions = ArrayList<Prediction>()
        val outputLabelMap = model!!.getLabelsMap()
        for (i in output.indices) {
            val percentage = round(output[i] * 10000) / 100
            val prediction = Prediction(outputLabelMap[i]!!, percentage)
            predictions.add(prediction)
        }
        predictions.sortByDescending { it.percentage }
        val highestPrediction = predictions[0]

        predictionBarChart?.changeData(predictions, highestPrediction.label)

        textView.text = highestPrediction.label

        predictionHistoryStorage?.let {
            val relativeTime = it.addPrediction(highestPrediction)
            predictionHistoryAdapter.addPrediction(
                highestPrediction,
                relativeTime,
                recyclerView
            )
        }
    }

    private fun predict() {
        lastPredictionTime = System.currentTimeMillis()
        if (window != null) {
            try {
                if (window!!.hasEnoughDataToCompileWindow()) {
                    model?.infer(arrayOf(window!!.compileWindowToArray()))?.let { updatePrediction(it[0]) }
                    window!!.clearValues()
                } else {
                    Toast.makeText(
                        context,
                        "Not enough data collected for prediction.",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            } catch (e: InMemoryWindow.SensorsOutOfSyncException) {
                activity.runOnUiThread {
                    stopDataCollection()
                    MessageDialog(
                        context,
                        "The following exception occurred:\n${e.message}\n\n" +
                                "To avoid this, sync the connected sensors on the " +
                                "connection screen before starting live prediction.",
                        "Sensors are out of sync"
                    )
                }
            }
        }
    }

    override fun onActivityCreated(activity: Activity) {

        this.activity = activity
        this.context = activity

        metadataStorage = XSensDotMetadataStorage(context)
        // Initializing prediction RV
        recyclerView = activity.findViewById(R.id.rv_prediction)
        predictionHistoryAdapter = PredictionHistoryAdapter(context, arrayListOf())
        recyclerView.adapter = predictionHistoryAdapter

        // Buttons and Timer
        timer = activity.findViewById(R.id.timer_predict_predict)
        textView = activity.findViewById(R.id.tv_predict_prediction)
        predictionButton = activity.findViewById(R.id.button_start_predict)
        progressBar = activity.findViewById(R.id.progressBar_nextPrediction_predictionFragment)

        predictionButton.setOnClickListener {
            if (!isRunning) {
                when (initModelFromCurrentUseCase()) {
                    ModelInitializationResult.SUCCESS -> {
                        val signatures = model?.signatureKeys
                        Log.d("PredictionScreen-onActivityCreated", signatures.toString())
                        togglePrediction()
                    }
                    ModelInitializationResult.MISSING_METADATA -> {
                        MessageDialog(
                            context,
                            message = context.getString(
                                R.string.missing_metadata_dialog_message
                            ),
                            title = context.getString(R.string.missing_metadata_dialog_title),
                            positiveButtonText = "Okay",
                            neutralButtonText = "Import default Model"
                        )
                    }
                    ModelInitializationResult.FILE_NOT_EXISTING -> {
                        currentUseCase.showNoModelFileExistingDialog(context)
                    }
                }
            } else {
                togglePrediction()
            }
        }

        toggleMotionLayout =
            activity.findViewById(R.id.motionLayout_predictionToggling_predictionFragment)

        mainHandler = Handler(Looper.getMainLooper())
    }

    override fun onActivityResumed() {}

    override fun onActivityWillDestroy() {
        // Nothing to do
    }

    override fun onConnectedDevicesChanged(deviceAddress: String, connected: Boolean) {

        numConnectedDevices = devices.getConnected().size

        if (isRunning && numConnectedDevices < model!!.getNumDevices()) {
            stopDataCollection()
            UIHelper.showAlert(context, "Connection to device(s) lost!")
        }
    }

    override fun onXsensDotDataChanged(deviceAddress: String, xsensDotData: XsensDotData) {
        val deviceTag = devices[deviceAddress]?.tag ?: return
        val deviceTagPrefix = XSensDotDeviceWithOfflineMetadata.extractTagPrefixFromTag(deviceTag)
        if (deviceTagPrefix != null) {
            window?.appendSensorData(deviceTagPrefix, xsensDotData)
        } else {
            togglePrediction()
            activity.runOnUiThread {
                Toast.makeText(
                    context,
                    "Could not extract prefixe for device tag $deviceTag",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    override fun onXsensDotOutputRateUpdate(deviceAddress: String, outputRate: Int) {
        // Nothing to do
    }

    override fun onUseCaseChanged(useCase: UseCase) {
        currentUseCase = useCase
    }

    enum class ModelInitializationResult { SUCCESS, FILE_NOT_EXISTING, MISSING_METADATA }

    private fun initModelFromCurrentUseCase(): ModelInitializationResult {
        if (!currentUseCase.getModelFile().exists()) {
            return ModelInitializationResult.FILE_NOT_EXISTING
        }
        try {
            initMetadataModel(currentUseCase.getModelFile())
        } catch (_: InvalidModelMetadata) {
            return ModelInitializationResult.MISSING_METADATA
        }
        return ModelInitializationResult.SUCCESS
    }

    @Throws(InvalidModelMetadata::class)
    private fun initMetadataModel(modelFile: File) {
        model = TFLiteModel(modelFile)
        predictionInterval = model!!.getWindowInMilliSeconds() + cushion // ms
    }

    companion object {
        const val cushion = 1000L // ms
    }
}
