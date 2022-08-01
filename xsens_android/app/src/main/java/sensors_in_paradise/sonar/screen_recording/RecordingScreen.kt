package sensors_in_paradise.sonar.screen_recording

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.view.TextureView
import android.media.MediaPlayer
import android.view.View
import android.widget.*
import androidx.camera.view.PreviewView
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.tabs.TabLayout
import com.xsens.dot.android.sdk.events.XsensDotData
import sensors_in_paradise.sonar.*
import sensors_in_paradise.sonar.screen_connection.ConnectionInterface
import sensors_in_paradise.sonar.XSENSArrayList
import sensors_in_paradise.sonar.screen_recording.camera.CameraManager
import sensors_in_paradise.sonar.util.PreferencesHelper
import sensors_in_paradise.sonar.use_cases.UseCase
import java.io.IOException

enum class RecordingTab(val position: Int) {
    RECORDINGS(0),
    ACTIVITIES(1),
    CAMERA(2)
}

class RecordingScreen(
    private val devices: XSENSArrayList,
    private val recordingsManager: RecordingDataManager,
    private val sensorOccupationInterface: SensorOccupationInterface?,
    private var currentUseCase: UseCase
) : ScreenInterface, ConnectionInterface,
    TabLayout.OnTabSelectedListener {
    private lateinit var context: Context
    private lateinit var timer: Chronometer

    private lateinit var recyclerViewRecordings: RecyclerView
    private lateinit var viewAnimator: ViewAnimator
    private lateinit var recordingsAdapter: RecordingsAdapter
    private lateinit var tabLayout: TabLayout
    private lateinit var activitiesCenterTV: TextView
    private lateinit var noRecordingsCenterTV: TextView
    private var activitiesTab: TabLayout.Tab? = null
    private var recordingsTab: TabLayout.Tab? = null
    private lateinit var cameraTab: TabLayout.Tab
    private var numConnectedDevices = 0

    private lateinit var loggingManager: LoggingManager
    private lateinit var activity: Activity
    private lateinit var cameraManager: CameraManager
    private lateinit var mediaPlayerStartSound: MediaPlayer
    private lateinit var mediaPlayerStopSound: MediaPlayer
    override fun onActivityCreated(activity: Activity) {
        this.context = activity
        this.activity = activity
        timer = activity.findViewById(R.id.timer)
        mediaPlayerStartSound = MediaPlayer.create(context, R.raw.start_beep)
        mediaPlayerStopSound = MediaPlayer.create(context, R.raw.stop_beep)
        recyclerViewRecordings = activity.findViewById(R.id.recyclerView_recordings_captureFragment)
        val linearLayoutManager = LinearLayoutManager(context)
        recyclerViewRecordings.layoutManager = linearLayoutManager
        recordingsAdapter = RecordingsAdapter(recordingsManager, context, currentUseCase)
        recordingsManager.addOnSizeChangedListener { updateNoRecordingsTVVisibility() }
        recyclerViewRecordings.adapter = recordingsAdapter
        viewAnimator = activity.findViewById(R.id.viewSwitcher_captureFragment)
        tabLayout = activity.findViewById(R.id.tabLayout_captureFragment)
        activitiesCenterTV = activity.findViewById(R.id.textView_no_activities_captureFragment)
        activitiesTab = tabLayout.getTabAt(1)
        recordingsTab = tabLayout.getTabAt(0)
        noRecordingsCenterTV = activity.findViewById(R.id.textView_noRecordings_captureFragment)
        cameraTab = tabLayout.newTab().apply {
            text = "Camera"
        }

        loggingManager = LoggingManager(
            context,
           currentUseCase,
            devices,
            activity.findViewById(R.id.buttonRecord),
            timer,
            activity.findViewById(R.id.tv_activity_captureFragment),
            activity.findViewById(R.id.tv_person_captureFragment),
            activity.findViewById(R.id.recyclerView_activities_captureFragment)
        )
        initializeLoggingManagerCallbacks()

        tabLayout.addOnTabSelectedListener(this)
        val previewView =
            activity.findViewById<PreviewView>(R.id.previewView_camera_captureFragment)
        val overlayView =
            activity.findViewById<TextureView>(R.id.surfaceView_camera_captureFragment)
        cameraManager =
            CameraManager(context, previewView, overlayView)
        updateNoRecordingsTVVisibility()
    }

    @Suppress("LongMethod", "ComplexMethod")
    private fun initializeLoggingManagerCallbacks() {
        loggingManager.setOnRecordingDone { recording ->
            if (tabLayout.selectedTabPosition != RecordingTab.CAMERA.position) {
                cameraManager.unbindPreview()
            }
            tabLayout.selectTab(recordingsTab)
            addRecordingToUI(
                recording
            )
            activitiesCenterTV.visibility = View.VISIBLE
        }

        loggingManager.setOnRecordingStarted {
            sensorOccupationInterface?.onSensorOccupationStatusChanged(true)
            if (PreferencesHelper.shouldPlaySoundOnRecordingStartAndStop(context)) {
                mediaPlayerStartSound.start()
            }

            if (!cameraManager.shouldShowVideo()) {
                tabLayout.selectTab(activitiesTab)
                activitiesCenterTV.visibility = View.GONE
            } else {
                // In case Image Analysis isn't possible, bitmap needs to be extracted from preview
                tabLayout.selectTab(cameraTab)
                cameraManager.bindPreview()
            }
            if (cameraManager.shouldCaptureVideo()) {
                val dir = GlobalValues.getVideoRecordingsTempDir(context)
                dir.mkdir()
                cameraManager.startRecordingVideo(
                    dir.resolve(
                        "before_" + System.currentTimeMillis().toString() + ".mp4"
                    )
                )
            }
            if (cameraManager.shouldRecordPose()) {
                val dir = GlobalValues.getPoseRecordingsTempDir(context)
                dir.mkdir()
                cameraManager.startRecordingPose(
                    dir.resolve(
                        "poseEstimation_" + System.currentTimeMillis().toString() + ".csv"
                    )
                )
            }
        }

        loggingManager.setAfterRecordingStarted { dir, metadata ->
            cameraManager.setOnVideoRecordingFinalized { videoCaptureStartTime, videoTempFile ->
                metadata.setVideoCaptureStartedTime(videoCaptureStartTime, true)
                try {
                    videoTempFile.copyTo(dir.resolve(Recording.VIDEO_CAPTURE_FILENAME))
                    videoTempFile.delete()
                    val recordingIndex =
                        recordingsManager.indexOfFirst { r -> r.dir == dir }
                    recordingsAdapter.notifyItemChanged(recordingIndex)
                } catch (e: IOException) {
                    e.printStackTrace()
                } catch (e: NoSuchFileException) {
                    e.printStackTrace()
                } catch (e: FileAlreadyExistsException) {
                    e.printStackTrace()
                } catch (e: IllegalArgumentException) {
                    e.printStackTrace()
                }
            }

            cameraManager.setOnPoseRecordingFinalized { poseCaptureStartTime, poseTempFile ->
                metadata.setPoseCaptureStartedTime(poseCaptureStartTime, true)
                try {
                    poseTempFile.copyTo(dir.resolve(Recording.POSE_CAPTURE_FILENAME))
                    poseTempFile.delete()
                    val recordingIndex =
                        recordingsManager.indexOfFirst { r -> r.dir == dir }
                    recordingsAdapter.notifyItemChanged(recordingIndex)
                } catch (e: IOException) {
                    e.printStackTrace()
                } catch (e: NoSuchFileException) {
                    e.printStackTrace()
                } catch (e: FileAlreadyExistsException) {
                    e.printStackTrace()
                } catch (e: IllegalArgumentException) {
                    e.printStackTrace()
                }
            }

            cameraManager.setOnVideoRecordingFailed { errorKey ->
                loggingManager.stopLoggingImmediately()
                Toast.makeText(context, "Video Recording failed: $errorKey", Toast.LENGTH_SHORT).show()
            }
        }

        loggingManager.setOnFinalizingRecording {
            if (PreferencesHelper.shouldPlaySoundOnRecordingStartAndStop(context)) {
                mediaPlayerStopSound.start()
            }
            cameraManager.stopRecordingVideo()
            cameraManager.stopRecordingPose()
            sensorOccupationInterface?.onSensorOccupationStatusChanged(false)
        }
    }

    private fun addRecordingToUI(recording: Recording) {
        recordingsManager.add(0, recording)
        recordingsAdapter.notifyItemInserted(0)
    }

    override fun onActivityResumed() {
        setCameraTabVisible(PreferencesHelper.shouldRecordWithCamera(context))
    }

    override fun onActivityWillDestroy() {
        loggingManager.stopLoggingImmediately()
    }

    override fun onConnectedDevicesChanged(deviceAddress: String, connected: Boolean) {
        numConnectedDevices = devices.getConnected().size
        loggingManager.handleConnectionStateChange(deviceAddress, connected)
    }

    override fun onXsensDotDataChanged(deviceAddress: String, xsensDotData: XsensDotData) {
        loggingManager.handleSensorDataUpdate(deviceAddress, xsensDotData)
    }

    override fun onXsensDotOutputRateUpdate(deviceAddress: String, outputRate: Int) {}

    override fun onTabSelected(tab: TabLayout.Tab?) {
        if (tab != null) {
            viewAnimator.displayedChild = tab.position
            if (tab == cameraTab) {
                cameraManager.bindPreview()
                cameraManager.clearPreview()
            }
        }
    }

    override fun onTabUnselected(tab: TabLayout.Tab?) {
        if (tab == cameraTab && !loggingManager.isRecording()) {
            cameraManager.unbindPreview()
        }
    }

    override fun onTabReselected(tab: TabLayout.Tab?) {}

    private fun setCameraTabVisible(visible: Boolean) {
        if (visible) {
            if (tabLayout.tabCount < 3) {
                tabLayout.addTab(cameraTab)
            }
        } else {
            if (tabLayout.tabCount > 2) {
                tabLayout.removeTab(cameraTab)
            }
        }
    }

    private fun updateNoRecordingsTVVisibility() {
        noRecordingsCenterTV.visibility = if (recordingsManager.isEmpty()) View.VISIBLE else View.INVISIBLE
    }

    @SuppressLint("NotifyDataSetChanged")
    override fun onUseCaseChanged(useCase: UseCase) {
        recordingsAdapter.notifyDataSetChanged()
        loggingManager.currentUseCase = useCase
        currentUseCase = useCase
        recordingsAdapter.currentUseCase = useCase
    }
}
