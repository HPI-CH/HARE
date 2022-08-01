package sensors_in_paradise.sonar.screen_recording

import android.content.Context
import android.os.SystemClock
import android.widget.Chronometer
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.xsens.dot.android.sdk.events.XsensDotData
import com.xsens.dot.android.sdk.models.XsensDotDevice
import com.xsens.dot.android.sdk.utils.XsensDotLogger
import sensors_in_paradise.sonar.GlobalValues
import sensors_in_paradise.sonar.R
import sensors_in_paradise.sonar.XSENSArrayList
import sensors_in_paradise.sonar.util.UIHelper
import sensors_in_paradise.sonar.use_cases.UseCase
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files
import java.time.LocalDateTime
import java.time.ZoneOffset

const val VALID_SENSOR_NUM = 5

@Suppress("LongParameterList")
class LoggingManager(
    val context: Context,
    var currentUseCase: UseCase,
    private val devices: XSENSArrayList,
    private val recordButton: MaterialButton,
    private val timer: Chronometer,
    private val labelTV: TextView,
    private val personTV: TextView,
    activitiesRV: RecyclerView
) {

    private var onRecordingDone: ((Recording) -> Unit)? = null
    private var onRecordingStarted: (() -> Unit)? = null
    private var afterRecordingStarted: ((dir: File, metadata: RecordingMetadataStorage) -> Unit)? =
        null
    private var onFinalizeRecording: (() -> Unit)? = null
    private val activitiesAdapter = ActivitiesAdapter(arrayListOf(), this)
    private var activeRecording: ActiveRecording? = null

    private var categoriesDialog: PersistentCategoriesDialog? = null
    private var changeCategoriesDialog: PersistentCategoriesDialog? = null
    private var openedTimestamp: Long = 0
    private var positionToChange: Int = 0

    init {
        activitiesRV.adapter = activitiesAdapter

        labelTV.setOnClickListener {
            showActivityDialog({ label, openedTimestamp ->
                updateSelectedLabel(label, openedTimestamp)
            })
        }
        personTV.setOnClickListener {
            showPersonDialog(this::updateSelectedPerson)
        }

        recordButton.setOnClickListener {
            toggleRecording()
        }
    }

    // /////////////////////////////
    // /////// UI UPDATES //////////
    // /////////////////////////////
    // Everything that deals with changing the UI to show the current recording state

    private fun updateSelectedPerson(name: String) {
        personTV.text = name
    }

    private fun updateSelectedLabel(label: String, openedTimestamp: Long) {
        labelTV.text = label
        val position = activeRecording?.addLabel(openedTimestamp, label) ?: 0
        activitiesAdapter.notifyItemInserted(position)
    }

    private fun updateRecordButton() {
        if (isRecording()) {
            recordButton.setIconResource(R.drawable.ic_baseline_stop_24)
        } else {
            recordButton.setIconResource(R.drawable.ic_baseline_play_arrow_24)
        }
    }

    private fun startUITimer() {
        timer.base = SystemClock.elapsedRealtime()
        timer.format = "%s"
        timer.start()
    }

    // /////////////////////////////
    // ////// STATE UPDATES ////////
    // /////////////////////////////

    /**
     * Checks if the changed sensor is still connected and if not stops the recording
     */
    fun handleConnectionStateChange(address: String, connected: Boolean) {
        if (activeRecording?.hasLoggerForAddress(address) == true && !connected) {
            devices.get(address)?.let {
                if (it.connectionState == XsensDotDevice.CONN_STATE_DISCONNECTED) {
                    UIHelper.showAlert(
                        context,
                        "The Device ${it.name} was disconnected!"
                    )
                    stopRecording()
                }
            }
        }
    }

    /**
     * Updates an existing logger with the new data, if it exists
     */
    fun handleSensorDataUpdate(address: String, xsensDotData: XsensDotData) {
        activeRecording?.getLoggerForAddress(address)?.update(xsensDotData)
    }

    // /////////////////////////////
    // //////// RECORDING //////////
    // /////////////////////////////

    /**
     * Stops the recording, if it is currently running or tries to start a new one if none is running
     *
     */
    private fun toggleRecording() {
        if (isRecording()) {
            stopRecording()
        } else if (isReadyToRecord()) {
            startRecording()
        }
    }

    private fun isReadyToRecord(): Boolean {
        val numConnected = devices.getConnected().size
        if (numConnected != VALID_SENSOR_NUM) {
            Toast.makeText(
                context,
                "Only $numConnected out of $VALID_SENSOR_NUM devices are connected!",
                Toast.LENGTH_LONG
            ).show()
        }
        return true
    }

    fun isRecording(): Boolean {
        return activeRecording?.isRunning() ?: false
    }

    /**
     * Updates the UI to represent a running recording, creates a new ActiveRecording and starts it.
     *
     * If no label is selected, the activity dialog is opened
     */
    private fun startRecording() {
        onRecordingStarted?.let { it1 -> it1() }

        startUITimer()

        activeRecording = ActiveRecording(context, devices, currentUseCase)
        activitiesAdapter.setAllActivities(activeRecording!!.labels)
        if (isLabelSelected()) {
            activeRecording!!.start(labelTV.text.toString())
        } else {
            activeRecording!!.start()
            showActivityDialog(
                cancelable = false,
                onSelected = this::updateSelectedLabel
            )
        }
        updateRecordButton()

        val (recordingDir, metadataStorage) = activeRecording!!.initializeRecordingDir()
        afterRecordingStarted?.let { it(recordingDir, metadataStorage) }
    }

    /**
     * Stops and deletes the current ActiveRecording WITHOUT saving it.
     */
    fun cancelLogging() {
        activeRecording?.stop()
        activitiesAdapter.unlinkActivitiesList()
        activeRecording = null
    }

    private fun stopRecording() {
        if (activeRecording == null) return

        val activeRecording = activeRecording!!
        activeRecording.stop()

        timer.stop()
        updateRecordButton()

        resolveMissingFields {
            val recordingFile = activeRecording.save(personTV.text.toString())
            recordingFile.resolve(GlobalValues.ACTIVE_RECORDING_FLAG_FILENAME).delete()
            activeRecording.recording?.let {
                onFinalizeRecording?.invoke()
                onRecordingDone?.invoke(it)
            }
            activitiesAdapter.unlinkActivitiesList()
            this.activeRecording = null
            labelTV.text = ""
        }
    }

    /**
     * Stops and saves the current recording immediately. Different to `stopLogging` the user is not
     * asked for input if values are missing.
     */
    fun stopLoggingImmediately() {
        if (!isLabelSelected()) {
            updateSelectedLabel(GlobalValues.NULL_ACTIVITY, 0)
        }
        if (!isPersonSelected()) {
            updateSelectedPerson(GlobalValues.UNKNOWN_PERSON)
        }
        stopRecording()
    }

    private fun resolveMissingFields(onAllResolved: () -> Unit) {
        if (!isLabelSelected()) {
            showActivityDialog(cancelable = false, onSelected = { label, _ ->
                updateSelectedLabel(label, 0)
                resolveMissingFields(onAllResolved)
            })
        } else if (!isPersonSelected()) {
            showPersonDialog(cancelable = false, onSelected = { person ->
                updateSelectedPerson(person)
                resolveMissingFields(onAllResolved)
            })
        } else {
            onAllResolved()
        }
    }

    private fun getCurrentOpenedTimestamp(): Long {
        return openedTimestamp
    }

    private fun getPositionToChange(): Int {
        return positionToChange
    }

    // /////////////////////////////
    // ///////// DIALOGS ///////////
    // /////////////////////////////

    private fun showActivityDialog(
        onSelected: (value: String, openedTimestamp: Long) -> Unit,
        cancelable: Boolean? = true
    ) {
        openedTimestamp = LocalDateTime.now().toSonarLong()

        if (categoriesDialog == null) {
            categoriesDialog = PersistentCategoriesDialog(
                context,
                "Select an activity label",
                currentUseCase.getActivityLabelsJSONFile(),
                callback = { value -> onSelected(value, getCurrentOpenedTimestamp()) },
            )
        }
        categoriesDialog?.show(cancelable ?: true)
    }

    private fun showChangeActivityDialog(
        onSelected: (value: String, openedTimestamp: Long) -> Unit,
        cancelable: Boolean? = true
    ) {
        if (changeCategoriesDialog == null) {
            changeCategoriesDialog = PersistentCategoriesDialog(
                context,
                "Select an activity label",
                currentUseCase.getActivityLabelsJSONFile(),
                callback = { value -> onSelected(value, getCurrentOpenedTimestamp()) },
            )
        }
        changeCategoriesDialog?.show(cancelable ?: true)
    }

    private fun showPersonDialog(
        onSelected: (value: String) -> Unit,
        cancelable: Boolean? = true
    ) {
        PersistentStringArrayDialog(
            context,
            "Select a Person",
            currentUseCase.getPeopleJSONFile(),
            defaultItem = GlobalValues.UNKNOWN_PERSON,
            callback = onSelected, cancelable = cancelable ?: true
        )
    }

    // /////////////////////////////
    // ///////// HELPERS ///////////
    // /////////////////////////////
    private fun isLabelSelected(): Boolean {
        val labelText = labelTV.text.toString()
        return labelText != ""
    }

    private fun isPersonSelected(): Boolean {
        val personText = personTV.text.toString()
        return personText != ""
    }

    fun setOnRecordingDone(onRecordingDone: (Recording) -> Unit) {
        this.onRecordingDone = onRecordingDone
    }

    fun setOnRecordingStarted(onRecordingStarted: () -> Unit) {
        this.onRecordingStarted = onRecordingStarted
    }

    fun setAfterRecordingStarted(afterRecordingStarted: (dir: File, metadata: RecordingMetadataStorage) -> Unit) {
        this.afterRecordingStarted = afterRecordingStarted
    }

    fun setOnFinalizingRecording(onFinalizeRecording: () -> Unit) {
        this.onFinalizeRecording = onFinalizeRecording
    }

    fun editActivityLabel(position: Int) {
        positionToChange = position
        showChangeActivityDialog(
            cancelable = true,
            onSelected = { value: String, _ ->
                this.activeRecording?.changeLabel(getPositionToChange(), value)
                this.activitiesAdapter.notifyItemChanged(getPositionToChange())
            }
        )
    }

    companion object {
        fun normalizeTimeStamp(time: LocalDateTime): Long {
            return time.toSonarLong()
        }
    }
}

/**
 * Represents a currently running recording, consisting of loggers for multiple sensors which write
 * to multiple temporary files.
 *
 * The recording can be started, stopped and saved and labels can be added during the recording
 */
class ActiveRecording(
    val context: Context,
    private val devices: XSENSArrayList,
    private val currentUseCase: UseCase
) {
    private val xsLoggers: ArrayList<XsensDotLogger> = ArrayList()
    var labels = ArrayList<RecordingMetadataStorage.LabelEntry>()
    private val sensorMacToTagMap = mutableMapOf<String, String>()
    private val tempSensorFiles = arrayListOf<Pair<String, File>>()

    private lateinit var recordingStartTime: LocalDateTime
    private var recordingEndTime: LocalDateTime? = null

    var recording: Recording? = null
        private set
    private var metadataStorage: RecordingMetadataStorage? = null

    fun start() {
        recordingStartTime = LocalDateTime.now()
        startXSensLoggers()
    }

    fun start(initialLabel: String) {
        start()
        addLabel(0, initialLabel)
    }

    fun stop() {
        recordingEndTime = LocalDateTime.now()
        stopXSensLoggers()
    }

    /** Saves recording and returns dir in which files were saved
     * */
    fun save(person: String): File {
        assert(recordingEndTime != null)
        val destFileDir =
            LogIOHelper.getOrCreateRecordingFileDir(recordingStartTime, currentUseCase)
        moveTempFiles(destFileDir, person, recordingEndTime!!.toSonarLong())
        return destFileDir
    }

    fun getLoggerForAddress(address: String): XsensDotLogger? {
        return xsLoggers.find { logger -> logger.filename.contains(address) }
    }

    fun hasLoggerForAddress(address: String): Boolean {
        return getLoggerForAddress(address) != null
    }

    fun isRunning(): Boolean {
        return recordingEndTime == null
    }

    /**
     * Adds the label to the list of labels and returns the position
     */
    fun addLabel(timestamp: Long, label: String): Int {
        // If its the first label, it has to start at recordingStartTime
        val actualTimestamp = if (labels.isEmpty()) recordingStartTime.toSonarLong() else timestamp
        labels.add(RecordingMetadataStorage.LabelEntry(actualTimestamp, label))
        return labels.size - 1
    }

    private fun startXSensLoggers() {
        // Start measuring for all connected sensors
        val fileDir = LogIOHelper.getTempRecordingDir(context)
        for (device in devices.getConnected()) {
            // Store the Tag and MAC address for saving the mapping to JSON
            sensorMacToTagMap[device.address] = device.tag
            startXSensLogger(device, fileDir)
        }
    }

    private fun startXSensLogger(device: XsensDotDevice, fileDir: File) {
        device.measurementMode = GlobalValues.MEASUREMENT_MODE
        device.startMeasuring()

        val file = LogIOHelper.getUniqueFile(fileDir, device.address)
        tempSensorFiles.add(Pair(device.address, file))
        xsLoggers.add(
            createXSensLogger(file, device)
        )
    }

    /**
     * Returns a correctly configured XsensDotLogger
     */
    private fun createXSensLogger(file: File, device: XsensDotDevice): XsensDotLogger {
        return XsensDotLogger(
            this.context,
            XsensDotLogger.TYPE_CSV,
            GlobalValues.MEASUREMENT_MODE,
            file.absolutePath,
            device.address,
            device.firmwareVersion,
            device.isSynced,
            device.currentOutputRate,
            null as String?,
            "appVersion",
            0
        )
    }

    private fun stopXSensLoggers() {
        for (logger in xsLoggers) {
            logger.stop()
        }
        for (device in devices.getConnected()) {
            device.stopMeasuring()
        }
    }

    private fun getOrCreateMetadataStorage(recordingDir: File): RecordingMetadataStorage {
        val metadataFile = recordingDir.resolve(GlobalValues.METADATA_JSON_FILENAME)
        val isMetadataActual = metadataFile == this.metadataStorage?.file

        if (this.metadataStorage == null || !isMetadataActual) {
            this.metadataStorage = RecordingMetadataStorage(metadataFile)
        }
        return this.metadataStorage!!
    }

    fun initializeRecordingDir(): Pair<File, RecordingMetadataStorage> {
        val destFileDir =
            LogIOHelper.getOrCreateRecordingFileDir(recordingStartTime, currentUseCase)
        val metadataStorage = getOrCreateMetadataStorage(destFileDir)
        val activeFlagFile = destFileDir.resolve(GlobalValues.ACTIVE_RECORDING_FLAG_FILENAME)
        activeFlagFile.createNewFile()
        return Pair(destFileDir, metadataStorage)
    }

    /**
     * Stores all temporary recording files and writes the metadata file.
     * Return the destination dir and metadata object
     */
    private fun moveTempFiles(destFileDir: File, person: String, recordingEndTime: Long) {
        val recordingFiles = tempSensorFiles

        LogIOHelper.moveTempFilesToFinalDirectory(destFileDir, recordingFiles)

        val metadataStorage = getOrCreateMetadataStorage(destFileDir)
        metadataStorage.setData(
            labels,
            recordingStartTime.toSonarLong(),
            recordingEndTime,
            person,
            sensorMacToTagMap
        )

        recording = Recording(destFileDir, metadataStorage)
    }

    fun changeLabel(position: Int, value: String) {
        labels[position].activity = value
    }
}

/**
 * Helper object for IO access to the recording files
 */
object LogIOHelper {
    fun moveTempFilesToFinalDirectory(
        destFileDir: File,
        recordings: ArrayList<Pair<String, File>>,
    ) {
        for ((deviceAddress, tempFile) in recordings) {
            val destFile = getRecordingFile(destFileDir, deviceAddress)
            Files.copy(tempFile.toPath(), FileOutputStream(destFile))
        }
    }

    fun getOrCreateRecordingFileDir(
        time: LocalDateTime,
        currentUseCase: UseCase
    ): File {
        val timeStr = time.toSonarString()

        val dir = currentUseCase.getRecordingsSubDir()
            .resolve(timeStr)

        if (!dir.exists()) {
            // assume this completes successful or the app crashes (but recording is lost anyway)
            dir.mkdirs()
        }

        return dir
    }

    private fun getRecordingFile(fileDir: File, deviceAddress: String): File {
        return fileDir.resolve("${escapeFilename(deviceAddress)}.csv")
    }

    private fun escapeFilename(name: String): String {
        return name.replace("\\W+".toRegex(), "-")
    }

    fun getUniqueFile(fileDir: File, deviceAddress: String): File {
        return fileDir.resolve("${System.currentTimeMillis()}_$deviceAddress.csv")
    }

    fun getTempRecordingDir(context: Context): File {
        val fileDir = GlobalValues.getSensorRecordingsTempDir(context)
        fileDir.mkdirs()
        return fileDir
    }
}

fun LocalDateTime.toSonarLong() = this.atZone(ZoneOffset.systemDefault()).toInstant().toEpochMilli()
fun LocalDateTime.toSonarString() = this.toSonarLong().toString()
