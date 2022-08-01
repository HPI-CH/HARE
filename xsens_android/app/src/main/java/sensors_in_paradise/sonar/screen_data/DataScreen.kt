package sensors_in_paradise.sonar.screen_data

import android.app.Activity
import android.content.Context
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.widget.SwitchCompat
import androidx.core.view.isVisible
import androidx.recyclerview.widget.RecyclerView
import com.github.mikephil.charting.charts.PieChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.PieEntry
import com.github.mikephil.charting.highlight.Highlight
import com.github.mikephil.charting.listener.OnChartValueSelectedListener
import sensors_in_paradise.sonar.R
import sensors_in_paradise.sonar.ScreenInterface
import sensors_in_paradise.sonar.machine_learning.TFLiteModel
import sensors_in_paradise.sonar.screen_recording.RecordingDataManager
import sensors_in_paradise.sonar.use_cases.UseCase
import sensors_in_paradise.sonar.util.dialogs.MessageDialog
import sensors_in_paradise.sonar.util.dialogs.file_explorer.FileExplorerDialog

class DataScreen(
    private val recordingsManager: RecordingDataManager,
    private var currentUseCase: UseCase
) : ScreenInterface {
    private lateinit var activitiesPieChart: PieChart
    private lateinit var activitiesPieChartPopulator: PieChartPopulator
    private lateinit var peoplePieChart: PieChart
    private lateinit var peoplePieChartPopulator: PieChartPopulator
    private lateinit var filterSwitch: SwitchCompat
    private lateinit var context: Context
    private lateinit var activity: Activity
    private lateinit var historyTV: TextView
    private lateinit var historyRV: RecyclerView
    private lateinit var dataHistoryAdapter: DataHistoryAdapter
    private lateinit var dataHistoryStorage: DataHistoryStorage
    private lateinit var trainBtn: Button
    private lateinit var predictBtn: Button
    private lateinit var exploreFilesBtn: Button
    override fun onActivityCreated(activity: Activity) {
        this.activity = activity
        this.context = activity
        activitiesPieChart = activity.findViewById(R.id.pieChart_availableData_dataFragment)
        activitiesPieChartPopulator = PieChartPopulator(context, activitiesPieChart)
        peoplePieChart = activity.findViewById(R.id.pieChart_availableDataPeople_dataFragment)
        peoplePieChartPopulator = PieChartPopulator(context, peoplePieChart)
        filterSwitch = activity.findViewById(R.id.switch_filterForTraining_dataFragment)
        historyTV = activity.findViewById(R.id.textView_historyHeading_dataFragment)
        historyRV = activity.findViewById(R.id.recyclerView_history_dataFragment)
        dataHistoryStorage = DataHistoryStorage(currentUseCase)
        dataHistoryAdapter = DataHistoryAdapter(dataHistoryStorage.getTrainingHistory())
        trainBtn = activity.findViewById(R.id.button_trainModel_dataFragment)
        predictBtn = activity.findViewById(R.id.button_predict_dataFragment)
        exploreFilesBtn = activity.findViewById(R.id.button_exploreFiles_dataFragment)
        historyRV.adapter = dataHistoryAdapter
        trainBtn.setOnClickListener {
            val model = attemptModelInit()
            model?.let {
                ModelTraining(context, recordingsManager, model) { recordingsTrainedOn ->
                    val activityDurations =
                        RecordingDataManager.getActivityDurations(recordingsTrainedOn)
                    val item = dataHistoryStorage.addTrainingOccasion(
                        currentUseCase.getRecordingsSubDir().name,
                        recordingsManager.getPeopleDurations(),
                        activityDurations
                    )
                    dataHistoryAdapter.trainingHistory.add(
                        0,
                        item
                    )
                    dataHistoryAdapter.notifyItemAdded(0)
                    historyRV.scrollToPosition(0)
                    historyTV.isVisible = true

                    model.save(
                        currentUseCase.getModelCheckpointsDir()
                            .resolve(item.timestamp.toString())
                    )
                }
            }
        }
        predictBtn.setOnClickListener {
            val model = attemptModelInit()
            model?.let {
                ModelPrediction(context, recordingsManager, model)
            }
        }

        activitiesPieChart.description.isEnabled = false

        exploreFilesBtn.setOnClickListener {
            FileExplorerDialog(
                context,
                currentUseCase.getRecordingsSubDir(),
                currentUseCase.getRelativePathOfRecordingsSubDir()
            )
        }
        peoplePieChart.description.isEnabled = false
        initPieChartListeners()
        filterSwitch.setOnCheckedChangeListener { _, _ -> populateAndAnimateCharts() }
    }

    private fun initPieChartListeners() {
        activitiesPieChart.setOnChartValueSelectedListener(object : OnChartValueSelectedListener {
            override fun onValueSelected(e: Entry?, h: Highlight?) {
                val pieEntry = e as PieEntry?
                peoplePieChartPopulator.populateAndAnimateChart(
                    recordingsManager.getPeopleDurations(
                        pieEntry?.label,
                        onlyUntrainedRecordings = filterSwitch.isChecked
                    )
                )
                peoplePieChart.isHighlightPerTapEnabled = false
            }

            override fun onNothingSelected() {
                peoplePieChartPopulator.populateAndAnimateChart(
                    recordingsManager.getPeopleDurations(onlyUntrainedRecordings = filterSwitch.isChecked)
                )
                peoplePieChart.isHighlightPerTapEnabled = true
            }
        })
        peoplePieChart.setOnChartValueSelectedListener(object : OnChartValueSelectedListener {
            override fun onValueSelected(e: Entry?, h: Highlight?) {
                val pieEntry = e as PieEntry?
                activitiesPieChartPopulator.populateAndAnimateChart(
                    recordingsManager.getActivityDurations(
                        pieEntry?.label,
                        onlyUntrainedRecordings = filterSwitch.isChecked
                    )
                )
                activitiesPieChart.isHighlightPerTapEnabled = false
            }

            override fun onNothingSelected() {
                activitiesPieChartPopulator.populateAndAnimateChart(
                    recordingsManager.getActivityDurations(onlyUntrainedRecordings = filterSwitch.isChecked)
                )
                activitiesPieChart.isHighlightPerTapEnabled = true
            }
        })
    }

    private fun attemptModelInit(): TFLiteModel? {
        val modelFile = currentUseCase.getModelFile()
        if (modelFile.exists()) {
            try {
                val model = TFLiteModel(modelFile)
                return model
            } catch (e: TFLiteModel.InvalidModelMetadata) {
                MessageDialog(
                    context,
                    "Could not load the TFLite model from current use case. " +
                            "The following exception was thrown:\n${e.message}",
                    "Failed loading model"
                )
            }
        } else {
            currentUseCase.showNoModelFileExistingDialog(context)
        }
        return null
    }

    override fun onScreenOpened() {
        populateAndAnimateCharts()
    }

    private fun populateAndAnimateCharts() {
        activitiesPieChartPopulator.populateAndAnimateChart(
            recordingsManager.getActivityDurations(onlyUntrainedRecordings = filterSwitch.isChecked)
        )
        peoplePieChartPopulator.populateAndAnimateChart(
            recordingsManager.getPeopleDurations(onlyUntrainedRecordings = filterSwitch.isChecked)
        )
    }

    override fun onUseCaseChanged(useCase: UseCase) {
        currentUseCase = useCase
        dataHistoryStorage = DataHistoryStorage(currentUseCase)
        dataHistoryAdapter.trainingHistory = dataHistoryStorage.getTrainingHistory()
        populateAndAnimateCharts()
    }
}
