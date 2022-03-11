package ro.ubb.mobile_app.detection

import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.detection.configuration.Configuration
import java.util.*
import kotlin.math.max
import kotlin.math.min

object Detector{
    private const val MAX_FONT_SIZE = 32F
    private const val MIN_FONT_SIZE = 15f

    private lateinit var detector: ObjectDetector
    private lateinit var configuration: Configuration

    fun isDetectorInitialized(): Boolean{
        return this::detector.isInitialized
    }

    fun setConfiguration(context: Context, configuration: Configuration){
        Log.v(TAG, "Detector settings:\n" +
                "name: ${configuration.modelName}\n" +
                "maxBoxes: ${configuration.maxNoBoxes}\n" +
                "minScore: ${configuration.scoreThreshold}\n" +
                "nmsIOU: ${configuration.nmsIouThreshold}\n")

        if(
            !this::configuration.isInitialized
            || configuration.modelName != this.configuration.modelName
            || configuration.maxNoBoxes != this.configuration.maxNoBoxes
            || configuration.scoreThreshold != this.configuration.scoreThreshold
        ) {
            Log.v(TAG, "reinitializing model...available cores: ${Runtime.getRuntime().availableProcessors()}")
            if(isDetectorInitialized())
                detector.close()
            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setMaxResults(configuration.maxNoBoxes)
                .setScoreThreshold(configuration.scoreThreshold / 100)
                .setBaseOptions(
                    BaseOptions.builder()
//                        .useGpu()
                        .setNumThreads(Runtime.getRuntime().availableProcessors())
                        .build()
                )
                .build()
            detector = ObjectDetector.createFromFileAndOptions(
                context,
                configuration.modelName,
                options
            )

        }
        this.configuration = configuration
    }

    private fun runDetection(bitmap: Bitmap): List<Detection> {
        var start = System.currentTimeMillis()
        val image = TensorImage.fromBitmap(bitmap)
        val results = detector.detect(image)
        Log.v(TAG, "Detection time: ${System.currentTimeMillis()-start}ms")

        Log.d(TAG, "Before NMS")
        logResults(results)
        start = System.currentTimeMillis()
        val nmsResults = nonMaximumSuppression(results)
        Log.v(TAG, "NMS time: ${System.currentTimeMillis()-start}ms")
        Log.d(TAG, "after NMS")
        logResults(nmsResults)
        return nmsResults
    }

    private fun intersectionOverUnion(box: RectF, otherBox: RectF): Float{
        val intersectWidth = max(min(box.right, otherBox.right) - max(box.left, otherBox.left), 0f)
        val intersectHeight = max(min(box.bottom, otherBox.bottom) - max(box.top, otherBox.top), 0f)

        val intersect = intersectHeight * intersectWidth
        val union = box.height() * box.width() + otherBox.height() * otherBox.width() - intersect

        return intersect / union
    }

    private fun nonMaximumSuppression(results: List<Detection>): List<Detection>{
        val newResults = LinkedList<Detection>()
        results.sortedByDescending { it.categories.first().score }
            .forEach{
                var maxIou = -1f
                val category = it.categories.first()
                for(result in newResults){
                    if(result.categories.first().label.equals(category.label)){
                        val currentIou = intersectionOverUnion(it.boundingBox, result.boundingBox)

                        if(maxIou < currentIou)
                            maxIou = currentIou
                    }
                }
                if(maxIou < configuration.nmsIouThreshold/100)
                    newResults.add(it)

            }
        return newResults
    }

    fun detect(bitmap: Bitmap): List<DetectionResult>{
        val results = runDetection(bitmap)
        return results.map {
            val category = it.categories.first()
            val label = category.label
            val score = category.score
            DetectionResult(it.boundingBox, label, score, category.index)
        }
    }

    fun imageWithBoxes(bitmap: Bitmap): Bitmap{
        val resultToDisplay = detect(bitmap)
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)
        drawDetectionResult(canvas, resultToDisplay)
        return outputBitmap
    }

    private fun logResults(results : List<Detection>) {
        Log.d(TAG, "#detections: ${results.size}")
        for ((i, obj) in results.withIndex()) {
            val box = obj.boundingBox

            Log.d(TAG, "Detected object: $i ")
            Log.d(TAG, "  boundingBox: (${box.left}, ${box.top}) - (${box.right},${box.bottom})")

            for ((j, category) in obj.categories.withIndex()) {
                Log.d(TAG, "    Label $j: ${category.label}")
                val confidence: Int = category.score.times(100).toInt()
                Log.d(TAG, "    Confidence: ${confidence}%")
            }
        }
    }

    private fun drawDetectionResult(
        canvas: Canvas,
        detectionResults: List<DetectionResult>
    ) {
        val pen = Paint()
        pen.textAlign = Paint.Align.LEFT

        detectionResults.forEach {
            when(it.classIndex){
                0-> pen.color = Color.RED
                1-> pen.color = Color.GREEN
                2-> pen.color = Color.BLUE
            }
            pen.strokeWidth = 3F
            pen.style = Paint.Style.STROKE
            val box = it.boundingBox
            canvas.drawRect(box, pen)


            val tagSize = Rect(0, 0, 0, 0)

            pen.style = Paint.Style.FILL_AND_STROKE
            pen.color = Color.MAGENTA
            pen.strokeWidth = 2F

            pen.textSize = MAX_FONT_SIZE

            val text = it.label + " " + "%,.2f".format(it.score * 100) + "%"

            pen.getTextBounds(text, 0, text.length, tagSize)
            val fontSize: Float = pen.textSize * box.width() / tagSize.width()

            if (fontSize < pen.textSize) pen.textSize = fontSize
            if(pen.textSize < MIN_FONT_SIZE) pen.textSize = MIN_FONT_SIZE
            canvas.drawText(
                text,
                box.left + 2f,
                box.top - 5f, pen
            )
        }
    }
}