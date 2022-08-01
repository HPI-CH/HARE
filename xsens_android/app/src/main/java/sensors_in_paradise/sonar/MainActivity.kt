package sensors_in_paradise.sonar

import android.content.Intent
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.view.Menu
import android.view.MenuInflater
import android.view.MenuItem
import android.widget.ViewAnimator
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate.*
import com.google.android.material.tabs.TabLayout
import com.google.firebase.crashlytics.FirebaseCrashlytics
import sensors_in_paradise.sonar.custom_views.stickman.StickmanDialog
import sensors_in_paradise.sonar.screen_connection.ConnectionInterface
import sensors_in_paradise.sonar.screen_connection.ConnectionScreen
import sensors_in_paradise.sonar.screen_prediction.PredictionScreen
import sensors_in_paradise.sonar.screen_recording.RecordingDataManager
import sensors_in_paradise.sonar.screen_recording.RecordingScreen
import sensors_in_paradise.sonar.screen_data.DataScreen
import sensors_in_paradise.sonar.uploader.DavCloudRecordingsUploader
import sensors_in_paradise.sonar.uploader.RecordingsUploaderDialog
import sensors_in_paradise.sonar.util.PreferencesHelper
import sensors_in_paradise.sonar.use_cases.UseCase
import sensors_in_paradise.sonar.use_cases.UseCaseDialog
import sensors_in_paradise.sonar.use_cases.UseCaseHandler

class MainActivity : AppCompatActivity(), TabLayout.OnTabSelectedListener, ConnectionInterface,
    SensorOccupationInterface {

    private lateinit var switcher: ViewAnimator
    private lateinit var tabLayout: TabLayout
    private lateinit var davCloudUploader: DavCloudRecordingsUploader
    private lateinit var recordingsManager: RecordingDataManager

    private val screenHandlers = ArrayList<ScreenInterface>()
    private lateinit var useCaseHandler: UseCaseHandler
    private val scannedDevices = XSENSArrayList()
    private lateinit var connectionScreen: ConnectionScreen
    private lateinit var sensorTrafficVisualizationHandler: SensorTrafficVisualizationHandler
    private lateinit var resetHeadingMi: MenuItem
    private lateinit var revertHeadingMi: MenuItem
    private lateinit var useCasesMi: MenuItem
    private lateinit var headingResetHandler: HeadingResetHandler
    private val tabIndexToScreenIndexMap = mutableMapOf(
        0 to 0,
        1 to 1,
        2 to 2,
        3 to 3
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        switcher = findViewById(R.id.switcher_activity_main)
        tabLayout = findViewById(R.id.tab_layout_activity_main)
        useCaseHandler = UseCaseHandler(this)
        recordingsManager = RecordingDataManager(
            useCaseHandler.getCurrentUseCase().getRecordingsSubDir()
        )

        supportActionBar?.subtitle = useCaseHandler.getCurrentUseCase().getDisplayInfo()
        useCaseHandler.setOnUseCaseChanged { useCase: UseCase ->
            recordingsManager.recordingsDir = useCase.getRecordingsSubDir()
            screenHandlers.forEach {
                it.onUseCaseChanged(
                    useCase
                )
            }
            supportActionBar?.subtitle = useCase.getDisplayInfo()
        }
        initClickListeners()

        connectionScreen = ConnectionScreen(scannedDevices)
        headingResetHandler = HeadingResetHandler(this, scannedDevices) { address ->
            runOnUiThread {
                connectionScreen.notifyItemChanged(address)
            }
        }
        connectionScreen.addConnectionInterface(headingResetHandler)
        screenHandlers.add(connectionScreen)

        val recordingScreen = RecordingScreen(
            scannedDevices,
            recordingsManager,
            this,
            useCaseHandler.getCurrentUseCase()
        )
        screenHandlers.add(recordingScreen)

        val dataScreen = DataScreen(recordingsManager, useCaseHandler.getCurrentUseCase())
        screenHandlers.add(dataScreen)

        val predictionScreen = PredictionScreen(useCaseHandler.getCurrentUseCase(), scannedDevices, this)
        screenHandlers.add(predictionScreen)

        connectionScreen.addConnectionInterface(recordingScreen)
        connectionScreen.addConnectionInterface(predictionScreen)
        connectionScreen.addConnectionInterface(this)

        val permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) {}
        screenHandlers.add(PermissionsHandler(permissionLauncher))

        for (handler in screenHandlers) {
            handler.onActivityCreated(this)
        }
        supportActionBar?.setBackgroundDrawable(ColorDrawable(getColor(R.color.colorPrimary)))

        davCloudUploader = DavCloudRecordingsUploader(this, RecordingDataManager(useCaseHandler.useCasesBaseDir))

        // Force crashlytics to be enabled (we might want to disable it in debug mode / ...)
        FirebaseCrashlytics.getInstance().setCrashlyticsCollectionEnabled(true)
    }

    override fun onResume() {
        super.onResume()
        var mode = MODE_NIGHT_FOLLOW_SYSTEM
        if (!PreferencesHelper.shouldFollowSystemTheme(this)) {
            mode = if (PreferencesHelper.shouldUseDarkMode(this)) MODE_NIGHT_YES else MODE_NIGHT_NO
        }
        setDefaultNightMode(mode)
        for (handler in screenHandlers) {
            handler.onActivityResumed()
        }
    }

    override fun onDestroy() {
        screenHandlers.forEach { handler -> handler.onActivityWillDestroy() }

        super.onDestroy()
    }

    private fun initClickListeners() {
        tabLayout.addOnTabSelectedListener(this)
    }

    override fun onTabSelected(tab: TabLayout.Tab?) {
        if (tab != null) {
            val screenIndex = tabIndexToScreenIndexMap[tab.position]!!
            switcher.displayedChild = screenIndex
            screenHandlers[screenIndex].onScreenOpened()
        }
    }

    override fun onTabUnselected(tab: TabLayout.Tab?) {
        if (tab != null) {
            val screenIndex = tabIndexToScreenIndexMap[tab.position]!!
            screenHandlers[screenIndex].onScreenClosed()
        }
    }

    override fun onTabReselected(tab: TabLayout.Tab?) {}

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        val inflater: MenuInflater = menuInflater
        inflater.inflate(R.menu.activity_main_menu, menu)
        sensorTrafficVisualizationHandler = SensorTrafficVisualizationHandler(
            this,
            scannedDevices,
            findViewById(R.id.sensorDataTrafficIndicator_captureFragment),
            findViewById(R.id.linearLayout_sensorOrientation_activityMain),
            menu.findItem(R.id.menuItem_orientation_activityMain)
        )
        connectionScreen.addConnectionInterface(
            sensorTrafficVisualizationHandler
        )
        useCasesMi = menu.findItem(R.id.menuItem_useCases_activityMain)
        resetHeadingMi = menu.findItem(R.id.menuItem_headingReset_activityMain)
        revertHeadingMi = menu.findItem(R.id.menuItem_headingRevert_activityMain)
        resetHeadingMi.isVisible = PreferencesHelper.shouldViewSensorHeadingMenuItems(this)
        revertHeadingMi.isVisible = PreferencesHelper.shouldViewSensorHeadingMenuItems(this)
        return true
    }

    fun onFileUploadMenuItemClicked(ignored: MenuItem) {
        RecordingsUploaderDialog(this, davCloudUploader).show()
    }

    fun onSettingsMenuItemClicked(ignored: MenuItem) {
        val intent = Intent(this, SettingsActivity::class.java)
        startActivity(intent)
    }

    fun onUseCasesMenuItemClicked(ignored: MenuItem) {
        UseCaseDialog(this, useCaseHandler)
    }

    fun onStickmanMenuItemClicked(ignored: MenuItem) {
        StickmanDialog(this)
    }

    fun onOrientationMenuItemClicked(mI: MenuItem) {
        mI.isChecked = !mI.isChecked
        sensorTrafficVisualizationHandler.setOrientationVisible(mI.isChecked)
    }

    fun onHeadingResetMenuItemClicked(ignored: MenuItem) {
        headingResetHandler.resetHeadings()
    }

    fun onHeadingRevertMenuItemClicked(ignored: MenuItem) {
        headingResetHandler.revertHeadings()
    }

    override fun onConnectedDevicesChanged(deviceAddress: String, connected: Boolean) {
        updateHeadingMenuItems()
    }

    private var areConnectedSensorsOccupied = false
    override fun onSensorOccupationStatusChanged(occupied: Boolean) {
        areConnectedSensorsOccupied = occupied
        updateHeadingMenuItems()
        useCasesMi.isEnabled = !occupied
    }

    private fun updateHeadingMenuItems() {
        val hasConnectedSensors = scannedDevices.getConnected().size > 0
        resetHeadingMi.isEnabled = !areConnectedSensorsOccupied && hasConnectedSensors
        revertHeadingMi.isEnabled = !areConnectedSensorsOccupied && hasConnectedSensors
    }
}
