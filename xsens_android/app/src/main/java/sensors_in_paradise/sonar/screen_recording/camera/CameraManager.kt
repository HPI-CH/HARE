package sensors_in_paradise.sonar.screen_recording.camera

import android.annotation.SuppressLint
import android.content.Context
import android.os.CountDownTimer
import android.util.Log
import android.util.Size
import android.view.TextureView
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.video.Recording

import androidx.camera.video.VideoCapture.withOutput
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.util.Consumer
import androidx.lifecycle.LifecycleOwner
import sensors_in_paradise.sonar.screen_recording.camera.pose_estimation.data.Device
import sensors_in_paradise.sonar.screen_recording.LoggingManager
import sensors_in_paradise.sonar.screen_recording.camera.pose_estimation.*
import sensors_in_paradise.sonar.screen_recording.camera.pose_estimation.data.Pose
import sensors_in_paradise.sonar.util.PreferencesHelper
import java.io.File
import java.time.LocalDateTime
import java.util.concurrent.Executors

class CameraManager(
    val context: Context,
    private val previewView: PreviewView,
    private val overlayView: TextureView
) :
    Consumer<VideoRecordEvent> {
    private var cameraProvider: ProcessCameraProvider? = null
    private var isPreviewBound = false
    private val preview: Preview = Preview.Builder()
        .build()
    private val videoCaptureExecutor = Executors.newFixedThreadPool(2)
    private val recorder = Recorder.Builder()
        .setExecutor(videoCaptureExecutor)
        .setQualitySelector(QualitySelector.from(PreferencesHelper.getCameraRecordingQuality(context)))
        .build()
    private val cameraSelector: CameraSelector = CameraSelector.Builder()
        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
        .build()
    private val videoCapture: VideoCapture<Recorder> = withOutput(recorder)
    private var videoRecording: Recording? = null

    private var imageProcessor: ImageProcessor? = null
    private var poseStorageManager: PoseEstimationStorageManager? = null
    private val imageAnalysisExecutor = Executors.newFixedThreadPool(2)

    @SuppressLint("UnsafeOptInUsageError")
    private val imageAnalysis = ImageAnalysis.Builder()
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .setTargetResolution(Size(ImageProcessor.INPUT_WIDTH, ImageProcessor.INPUT_HEIGHT))
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()
        .apply {
            setAnalyzer(imageAnalysisExecutor) { imageProxy ->
                val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                imageProxy.image?.let {
                    imageProcessor?.processImage(
                        it, overlayView, rotationDegrees == 90
                    )
                }
                imageProxy.close()
            }
        }
    private var timer: CountDownTimer? = null

    init {
        ProcessCameraProvider.getInstance(context).apply {
            addListener({
                cameraProvider = this.get()
            }, ContextCompat.getMainExecutor(context))
        }
    }

    private fun getTaskTimerObject(delay: Long, interval: Long): CountDownTimer {
        return object : CountDownTimer(delay, interval) {
            override fun onTick(millisUntilFinished: Long) {
                previewView.bitmap?.let { bm ->
                    imageProcessor?.processImage(
                        bm, overlayView, false
                    )
                }
            }

            override fun onFinish() {}
        }
    }

    fun shouldShowVideo(): Boolean {
        return PreferencesHelper.shouldRecordWithCamera(context)
    }

    fun bindPreview(): Boolean {
        if (cameraProvider != null && !isPreviewBound) {
            preview.setSurfaceProvider(previewView.surfaceProvider)
            val camera =
                cameraProvider?.bindToLifecycle(context as LifecycleOwner, cameraSelector, preview)
            isPreviewBound = camera != null
            Log.d("CameraManager", "Binding preview successful: $isPreviewBound")
        }
        return isPreviewBound
    }

    fun unbindPreview() {
        Log.d("CameraManager", "Unbinding preview")

        if (isPreviewBound) {
            isPreviewBound = false
            cameraProvider?.unbind(preview)
        }
    }

    fun clearPreview() {
        imageProcessor?.clearView(overlayView)
    }

    private var isCaptureBound = false
    private fun bindVideoCapture(): Boolean {
        if (cameraProvider != null && !isCaptureBound) {
            val camera =
                cameraProvider?.bindToLifecycle(
                    context as LifecycleOwner,
                    cameraSelector,
                    videoCapture
                )
            isCaptureBound = camera != null
            Log.d("CameraManager", "Binding video capture successful: $isCaptureBound")
        }
        return isCaptureBound
    }

    private fun unbindVideoCapture() {
        Log.d("CameraManager", "Unbinding video capture...")
        isCaptureBound = false
        cameraProvider?.unbind(videoCapture)
    }

    private var videoFile: File? = null
    private var videoStartTime: Long = 0L

    fun startRecordingVideo(outputFile: File): Recording {
        if (!isCaptureBound) {
            bindVideoCapture()
        }
        videoFile = outputFile
        val outputOptions = FileOutputOptions.Builder(outputFile).build()

        val recording = videoCapture.output
            .prepareRecording(context, outputOptions)
            .start(ContextCompat.getMainExecutor(context), this)
        videoRecording = recording
        return recording
    }

    private var onVideoRecordingFinalized: ((captureStartTime: Long, tempFile: File) -> Unit)? =
        null

    fun setOnVideoRecordingFinalized(onRecordingFinalized: ((captureStartTime: Long, tempFile: File) -> Unit)? = null) {
        this.onVideoRecordingFinalized = onRecordingFinalized
    }

    private var onVideoRecordingFailed: ((errorKey: String) -> Unit)? = null

    fun setOnVideoRecordingFailed(onVideoRecordingFailed: ((errorKey: String) -> Unit)) {
        this.onVideoRecordingFailed = onVideoRecordingFailed
    }

    /** Stops the recording and returns the UNIX timestamp of
     * when the recording did actually start and the file where it's stored
     * */
    fun stopRecordingVideo() {
        if (videoRecording == null) {
            return
        }
        videoRecording?.close()
        unbindVideoCapture()
    }

    fun shouldCaptureVideo(): Boolean {
        return shouldShowVideo() && PreferencesHelper.shouldStoreRawCameraRecordings(context)
    }

    override fun accept(t: VideoRecordEvent?) {
        when (t) {
            is VideoRecordEvent.Start -> {
                videoStartTime = LoggingManager.normalizeTimeStamp(LocalDateTime.now())
                Log.d("CameraManager", "Video Recording started")
            }
            is VideoRecordEvent.Finalize -> {
                onVideoRecordingFinalized?.let { it(videoStartTime, videoFile!!) }

                val finalizeError = t.error
                if (finalizeError != VideoRecordEvent.Finalize.ERROR_NONE) {
                    val errorKey =
                        if (finalizeError == VideoRecordEvent.Finalize.ERROR_FILE_SIZE_LIMIT_REACHED) {
                            "max file size reached"
                        } else "ERROR $finalizeError"
                    onVideoRecordingFailed?.invoke(errorKey)
                    Log.d("CameraManager", "Video Recording failed: $errorKey")
                }
                Log.d("CameraManager", "Video Recording finalized")
            }
            is VideoRecordEvent.Pause -> {
                Log.d("CameraManager", "Video Recording paused")
            }
            is VideoRecordEvent.Status -> {
                Log.d("CameraManager", "Video Recording status updated")
            }
        }
    }

    private var isAnalyzerBound = false

    private fun bindImageAnalyzer(): Boolean {
        if (cameraProvider != null && !isAnalyzerBound) {
            val camera =
                cameraProvider?.bindToLifecycle(
                    context as LifecycleOwner,
                    cameraSelector,
                    imageAnalysis
                )
            isAnalyzerBound = camera != null
            Log.d("CameraManager", "Binding image analyzer successful: $isAnalyzerBound")
        }
        return isAnalyzerBound
    }

    private fun unbindImageAnalyzer() {
        Log.d("CameraManager", "Unbinding image analyzer...")
        isAnalyzerBound = false
        cameraProvider?.unbind(imageAnalysis)
    }

    private var poseStartTime: Long? = null

    fun startRecordingPose(outputFile: File) {
        // Video Capture and Image Analysis use cases are not combinable
        if (shouldCaptureVideo()) {
            timer =
                getTaskTimerObject(Long.MAX_VALUE, 1000L / ImageProcessor.POSE_ESTIMATION_FREQUENCY)
            timer!!.start()
        } else if (!isAnalyzerBound) {
            bindImageAnalyzer()
        }

        poseStorageManager = if (poseStorageManager == null) PoseEstimationStorageManager(outputFile)
            else poseStorageManager!!.reset(outputFile)

        val useHandPoseEstimation = PreferencesHelper.shouldUseHandPoseEstimation(context)
        if (useHandPoseEstimation) {
            val localDateTime = LocalDateTime.now()
            poseStorageManager!!.writeHeader(
                localDateTime,
                Pair(ImageProcessor.INPUT_WIDTH, ImageProcessor.INPUT_HEIGHT),
                HandDetector.MODEL_NAME,
                3,
                Pose.HandPose
            )
            poseStartTime = LoggingManager.normalizeTimeStamp(localDateTime)

            val isDetectorActual =
                imageProcessor?.bodyPoseDetector == null &&
                imageProcessor?.handPoseDetector != null

            if (imageProcessor == null || !isDetectorActual) {
                imageProcessor = ImageProcessor(
                    context, poseStorageManager!!,
                    handPoseDetector = HandDetector(context)
                )
            }
        } else {
            val poseModel = PreferencesHelper.getPoseEstimationModel(context)
            val localDateTime = LocalDateTime.now()
            poseStorageManager!!.writeHeader(
                localDateTime,
                Pair(ImageProcessor.INPUT_WIDTH, ImageProcessor.INPUT_HEIGHT),
                poseModel.toString(),
                2,
                Pose.BodyPose
            )
            poseStartTime = LoggingManager.normalizeTimeStamp(localDateTime)

            val isDetectorActual =
                imageProcessor?.bodyPoseDetector != null &&
                imageProcessor?.handPoseDetector == null
            val isPoseModelActual =
                imageProcessor?.bodyPoseDetector?.modelType == poseModel

            if (imageProcessor == null || !isDetectorActual || !isPoseModelActual) {
                imageProcessor = ImageProcessor(
                    context, poseStorageManager!!,
                    bodyPoseDetector = createPoseEstimator(poseModel)
                )
            }
        }
    }

    private var onPoseRecordingFinalized: ((captureStartTime: Long, tempFile: File) -> Unit)? =
        null

    fun setOnPoseRecordingFinalized(onRecordingFinalized: ((captureStartTime: Long, tempFile: File) -> Unit)? = null) {
        this.onPoseRecordingFinalized = onRecordingFinalized
    }

    fun stopRecordingPose() {
        timer?.let {
            it.cancel()
            timer = null
        }
        if (isAnalyzerBound) {
            unbindImageAnalyzer()
        }

        if (imageProcessor != null && poseStorageManager != null) {
            clearPreview()
            poseStorageManager?.closeFile()
            onPoseRecordingFinalized?.invoke(poseStartTime ?: 0L, poseStorageManager!!.csvFile)
        }
    }

    fun shouldRecordPose(): Boolean {
        return shouldShowVideo() && PreferencesHelper.shouldStorePoseEstimation(context)
    }

    private fun createPoseEstimator(modelType: ModelType): MoveNet {
        val targetDevice = Device.GPU

        return MoveNet.create(context, targetDevice, modelType)
    }
}
