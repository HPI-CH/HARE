package sensors_in_paradise.sonar.custom_views.stickman

import android.app.AlertDialog
import android.content.Context
import android.view.LayoutInflater
import android.widget.SeekBar
import android.widget.SeekBar.OnSeekBarChangeListener
import sensors_in_paradise.sonar.R
import sensors_in_paradise.sonar.custom_views.stickman.object3d.Sensor3D

class StickmanDialog(context: Context) : OnSeekBarChangeListener {
    private var dialog: AlertDialog
    private val coordinateSystem3D = Sensor3D().apply {
        drawVertexPositionsForDebugging = true
    }
    private var xSlider: SeekBar
    private var ySlider: SeekBar
    private var zSlider: SeekBar

    init {

        val builder: AlertDialog.Builder = AlertDialog.Builder(context)
        builder.setTitle("Stickman")

        val root = LayoutInflater.from(context).inflate(R.layout.stickman_dialog, null)
        val stickmanView = root.findViewById<Render3DView>(R.id.stickmanView)
        xSlider = root.findViewById(R.id.slider_xEuler_stickmanDialog)
        ySlider = root.findViewById(R.id.slider_yEuler_stickmanDialog)
        zSlider = root.findViewById(R.id.slider_zEuler_stickmanDialog)
        stickmanView.apply {
            addObject3D(coordinateSystem3D)
        }

        stickmanView.camera.center.apply {
            x = 0f
            y = 0.5f
            z = 0f
        }
        stickmanView.camera.eye.apply {
            x = 0f
            y = 1f
            z = -2f
        }
        stickmanView.camera.up.apply {
            x = 0f
            y = 1f
            z = 0f
        }
        stickmanView.camera.notifyCameraChanged()
        stickmanView.onObjectChanged()
        stickmanView.enableYRotation = true
        stickmanView.showFPS = true

        xSlider.setOnSeekBarChangeListener(this)
        ySlider.setOnSeekBarChangeListener(this)
        zSlider.setOnSeekBarChangeListener(this)

        builder.setView(root)
        builder.setNegativeButton(
            "Cancel"
        ) { dialog, _ -> dialog.cancel() }

        dialog = builder.create()
        dialog.show()
    }

    override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
        coordinateSystem3D.resetToDefaultState(shouldNotifyThatVerticesChanged = false)
        coordinateSystem3D.rotateEuler(
            xSlider.progress.toFloat(),
            ySlider.progress.toFloat(),
            zSlider.progress.toFloat()
        )
    }

    override fun onStartTrackingTouch(seekBar: SeekBar?) {
    }

    override fun onStopTrackingTouch(seekBar: SeekBar?) {
    }
}
