package sensors_in_paradise.sonar.machine_learning

import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.metadata.MetadataExtractor
import sensors_in_paradise.sonar.custom_views.confusion_matrix.ConfusionMatrix
import java.io.File
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import kotlin.Array

class TFLiteModel @Throws(InvalidModelMetadata::class) constructor(tfLiteModelFile: File) {
    class InvalidModelMetadata(message: String) : Exception(message)

    val windowSize: Int
    private val frequency = 60
    private val output: FloatBuffer
    private val interpreter: Interpreter
    private val extractor: MetadataExtractor
    private val labels: ArrayList<String>
    private val features: Array<String>
    val signatureKeys: Array<String>

    init {
        val buffer = ByteBuffer.allocate(tfLiteModelFile.readBytes().size)
        buffer.put(tfLiteModelFile.readBytes())
        buffer.rewind()
        extractor = MetadataExtractor(buffer)
        if (!hasMetadata()) {
            throw InvalidModelMetadata("The model ${tfLiteModelFile.name} does not have any metadata")
        }
        val inputs = extractor.getInputTensorShape(0)
        interpreter = Interpreter(tfLiteModelFile).apply {
            resizeInput(0, inputs)
        }
        signatureKeys = interpreter.signatureKeys
        features = readFeatureFile().split("\n").filter { it.isNotEmpty() }.toTypedArray()
        labels = ArrayList(readLabelsFile().split("\n").filter { it.isNotEmpty() })
        output = FloatBuffer.allocate(labels.size)
        windowSize = extractor.getInputTensorShape(0)[1]
    }

    private fun hasMetadata(): Boolean {
        return extractor.hasMetadata()
    }

    fun close() {
        interpreter.close()
    }

    private fun readFeatureFile(): String {
        val inputStream = extractor.getAssociatedFile("features.txt")
        return inputStream.bufferedReader().use { it.readText() }
    }

    private fun readLabelsFile(): String {
        val inputStream = extractor.getAssociatedFile("labels.txt")
        return inputStream.bufferedReader().use { it.readText() }
    }

    fun getLabelsMap(): Map<Int, String> {
        return labels.mapIndexed { index, label ->
            index to label
        }.toMap()
    }

    fun getLabels(): List<String> {
        return labels
    }

    fun getNumOutputClasses(): Int {
        return labels.size
    }

    fun getDeviceTags(): Array<String> {
        return features.map { it.substringAfterLast("_") }.distinct().filter { it != "" }
            .toTypedArray()
    }

    fun getNumDevices(): Int {
        return getDeviceTags().size
    }

    fun getFeaturesToPredict(): Array<String> {
        return features
    }

    fun getWindowInMilliSeconds(): Long {
        return 1000L * (windowSize / frequency)
    }

    fun infer(windows: Array<Array<FloatArray>>): Array<FloatArray> {
        val inputs = HashMap<String, Any>()
        inputs["input_window"] = windows
        val outputs = HashMap<String, Any>()
        Log.d("TFLiteModel-runInfer", "Output buffer capacity: ${labels.size * windows.size}")
        val output = Array(windows.size) { FloatArray(labels.size) }
        outputs["output"] = output
        interpreter.runSignature(inputs, outputs, "infer")
        return output
    }

    fun evaluate(
        batches: Collection<Batch>,
        progressCallback: ((Int, Int) -> Unit)? = null
    ): Pair<Float, ConfusionMatrix> {
        val confusionMatrix = ConfusionMatrix(labels.toTypedArray())
        var accuracy = 0f
        for ((index, batch) in batches.withIndex()) {
            val interBatchProgress = (index * 100) / batches.size
            val batchAccuracy = evaluate(batch, confusionMatrix) { intraBatchProgress ->
                progressCallback?.let {
                    it(interBatchProgress, intraBatchProgress)
                }
            }
            accuracy += batchAccuracy / batches.size
        }

        return Pair(accuracy, confusionMatrix)
    }

    fun evaluate(
        batch: Batch,
        confusionMatrix: ConfusionMatrix? = null,
        progressCallback: ((Int) -> Unit)? = null
    ): Float {

        val (windows, labels) = batch.compile(this, true, progressCallback)
        val prediction = infer(windows)
        val accuracy = getAccuracyFromPredictions(prediction, labels)
        val predictedLabels = prediction.map { convertPredictionToLabel(it) }
        val actualLabels = labels.map { convertPredictionToLabel(it) }

        confusionMatrix?.addPredictions(predictedLabels.toTypedArray(), actualLabels.toTypedArray())
        return accuracy
    }

    fun train(batch: Batch, progressCallback: ((Int) -> Unit)? = null): Float {
        val (windows, labels) = batch.compile(this, false, progressCallback)
        return train(windows, labels)
    }

    private fun toPercentage(v: Int, max: Int, min: Int = 0): Int {
        return (v * 100) / (max - min)
    }

    /*Returns an array of losses with one entry per epoch*/
    fun train(
        batches: Collection<Batch>,
        epochs: Int = 1,
        progressCallback: ((Int, Int, Int) -> Unit)? = null
    ): FloatArray {
        val losses = FloatArray(epochs)
        for (e in 0 until epochs) {
            var epochLoss = 0f
            for ((batchIndex, batch) in batches.withIndex()) {
                val batchLoss = train(batch) { intraBatchProgress ->
                    progressCallback?.let {
                        it(
                            toPercentage(e, epochs),
                            toPercentage(batchIndex, batches.size),
                            intraBatchProgress
                        )
                    }
                }
                epochLoss += batchLoss / batches.size
            }
            losses[e] = epochLoss
        }
        return losses
    }

    fun train(windows: Array<Array<FloatArray>>, labels: Array<FloatArray>): Float {
        if (windows.isEmpty()) {
            throw java.lang.IllegalArgumentException("Can't run training on an empty windows array")
        }
        val numFeatures = features.size
        if (windows.size != labels.size) {
            throw java.lang.IllegalArgumentException(
                "Window size ${windows.size} is different to " +
                        "number of labels ${labels.size}"
            )
        }
        if (windows[0].size != windowSize) {
            throw java.lang.IllegalArgumentException(
                "Size of provided windows ${windows[0].size} " +
                        "is different to expected window size: $windowSize"
            )
        }
        if (windows[0][0].size != numFeatures) {
            throw java.lang.IllegalArgumentException(
                "Number of provided features " +
                        "${windows[0][0].size} inside the windows is different to expected" +
                        " number of features: $numFeatures"
            )
        }
        val batchSize = windows.size
        Log.d("Test-TFLiteModel - runTraining", "Running training on batch size $batchSize")
        val inputs = HashMap<String, Any>()
        inputs["input_windows"] = windows
        inputs["labels"] = labels
        val outputs = HashMap<String, Any>()
        val loss = FloatBuffer.allocate(1)
        loss.rewind()
        outputs["loss"] = loss
        interpreter.runSignature(inputs, outputs, "train")
        return loss.array()[0]
    }

    fun save(file: File) {
        val inputs = HashMap<String, Any>()
        val outputs = HashMap<String, Any>()
        val checkpointPath = file.absolutePath
        inputs["checkpoint_path"] = checkpointPath
        interpreter.runSignature(inputs, outputs, "save")
    }

    fun runRestore(file: File) {
        val inputs = HashMap<String, Any>()
        val outputs = HashMap<String, Any>()
        val checkpointPath = file.absolutePath
        inputs["checkpoint_path"] = checkpointPath
        interpreter.runSignature(inputs, outputs, "restore")
    }

    @Throws(IllegalArgumentException::class)
    fun convertPredictionToLabel(prediction: FloatArray): String {
        val labels = getLabelsMap()
        if (labels.size != prediction.size) {
            throw IllegalArgumentException(
                "Number of items in prediction ${prediction.size} " +
                        "is different to number of labels in this model ${labels.size}"
            )
        }
        val index = prediction.indices.maxByOrNull {
            prediction[it]
        }!!
        return labels[index]!!
    }

    @Throws(IllegalArgumentException::class)
    fun convertLabelToOneHotEncoding(label: String): FloatArray {
        val labels = getLabelsMap()
        if (!labels.containsValue(label)) {
            throw IllegalArgumentException(
                "Label $label " +
                        "is not in labels of model: $labels"
            )
        }
        val index = labels.filter { label == it.value }.keys.first()

        val result = FloatArray(labels.size)
        result[index] = 1f
        return result
    }

    fun getAccuracyFromPredictions(
        predictions: Array<FloatArray>,
        actualLabels: Array<String>
    ): Float {
        if (predictions.size != actualLabels.size) {
            throw IllegalArgumentException(
                "Size of predictions (${predictions.size}) is " +
                        "unequals to size of actualLabels (${actualLabels.size})"
            )
        }
        var correctPredictions = 0f
        for (i in predictions.indices) {
            val prediction = predictions[i]
            val label = actualLabels[i]
            if (convertPredictionToLabel(prediction) == label) {
                correctPredictions++
            }
        }
        return correctPredictions / predictions.size
    }

    fun getAccuracyFromPredictions(
        predictions: Array<FloatArray>,
        actualLabels: Array<FloatArray>
    ): Float {
        if (predictions.size != actualLabels.size) {
            throw IllegalArgumentException(
                "Size of predictions (${predictions.size}) " +
                        "is unequals to size of actualLabels (${actualLabels.size})"
            )
        }
        var correctPredictions = 0f
        for (i in predictions.indices) {
            val prediction = predictions[i]
            val label = actualLabels[i]
            if (convertPredictionToLabel(prediction) == convertPredictionToLabel(label)) {
                correctPredictions++
            }
        }
        return correctPredictions / predictions.size
    }
}
