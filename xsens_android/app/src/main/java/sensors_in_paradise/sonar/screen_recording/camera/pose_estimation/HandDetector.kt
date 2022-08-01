package sensors_in_paradise.sonar.screen_recording.camera.pose_estimation

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.solutioncore.ResultListener
import com.google.mediapipe.solutions.hands.Hands
import com.google.mediapipe.solutions.hands.HandsOptions
import com.google.mediapipe.solutions.hands.HandsResult

class HandDetector(context: Context) {
    private val hands: Hands

    init {
        val handsOptions = HandsOptions.builder()
            .setMaxNumHands(2)
            .setStaticImageMode(true)
            .setRunOnGpu(true)
            .build()
        hands = Hands(context, handsOptions)
    }

    fun estimatePose(bitmap: Bitmap, resultListener: ResultListener<HandsResult>) {
        hands.setResultListener(resultListener)
        hands.send(bitmap)
    }

    companion object {
        const val MODEL_NAME = "HandLandmark"
    }
}
