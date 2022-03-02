package ro.ubb.mobile_app.detection

import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.detection.configuration.Configuration
import java.util.*
import kotlin.math.max
import kotlin.math.min

object Detector{
    private const val MAX_FONT_SIZE = 96F

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
            Log.v(TAG, "reinitializing model")
            if(isDetectorInitialized())
                detector.close()
            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setMaxResults(configuration.maxNoBoxes)
                .setScoreThreshold(configuration.scoreThreshold / 100)
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

        Log.v(TAG, "Before NMS")
        debugPrint(results)
        start = System.currentTimeMillis()
        val nmsResults = nonMaximumSupression(results)
        Log.v(TAG, "NMS time: ${System.currentTimeMillis()-start}ms")
        Log.v(TAG, "after NMS")
        debugPrint(nmsResults)
        return nmsResults
    }

    private fun intersectionOverUnion(box: RectF, otherBox: RectF): Float{
        val intersectWidth = max(min(box.right, otherBox.right) - max(box.left, otherBox.left), 0f)
        val intersectHeight = max(min(box.bottom, otherBox.bottom) - max(box.top, otherBox.top), 0f)

        val intersect = intersectHeight * intersectWidth
        val union = box.height() * box.width() + otherBox.height() * otherBox.width() - intersect

        return intersect / union
    }

    private fun nonMaximumSupression(results: List<Detection>): List<Detection>{
        val newResults = LinkedList<Detection>()
        results.sortedByDescending { it.categories.first().score }
            .forEach{
                var maxIou = -1f
                val category = it.categories.first()
                for(result in newResults){
                    if(result.categories.first().label.equals(category.label)){
                        val currentIou = intersectionOverUnion(it.boundingBox, result.boundingBox)
                        Log.v(TAG, "$currentIou - iou between ${results.indexOf(it)} and ${results.indexOf(result)}")

                        if(maxIou < currentIou)
                            maxIou = currentIou
                    }
                }
                if(maxIou < configuration.nmsIouThreshold/100)
                    newResults.add(it)

            }
        return newResults
    }

    fun imageWithBoxes(bitmap: Bitmap): Bitmap{
        val results = runDetection(bitmap)
        val resultToDisplay = results.map {
            val category = it.categories.first()
            val text = "${category.label}, ${category.score.times(100).toInt()}%"
            DetectionResult(it.boundingBox, text, category.index)
        }
        return drawDetectionResult(bitmap, resultToDisplay)
    }

    private fun debugPrint(results : List<Detection>) {
        Log.v(TAG, "#detections: ${results.size}")
        for ((i, obj) in results.withIndex()) {
            val box = obj.boundingBox

            Log.v(TAG, "Detected object: $i ")
            Log.v(TAG, "  boundingBox: (${box.left}, ${box.top}) - (${box.right},${box.bottom})")

            for ((j, category) in obj.categories.withIndex()) {
                Log.v(TAG, "    Label $j: ${category.label}")
                val confidence: Int = category.score.times(100).toInt()
                Log.v(TAG, "    Confidence: ${confidence}%")
            }
        }
    }

    private fun drawDetectionResult(
        bitmap: Bitmap,
        detectionResults: List<DetectionResult>
    ): Bitmap {
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)
        val pen = Paint()
        pen.textAlign = Paint.Align.LEFT

        detectionResults.forEach {
            // draw bounding box
            when(it.classIndex){
                0-> pen.color = Color.RED
                1-> pen.color = Color.GREEN
                2-> pen.color = Color.BLUE
            }
            pen.strokeWidth = 8F
            pen.style = Paint.Style.STROKE
            val box = it.boundingBox
            canvas.drawRect(box, pen)


            val tagSize = Rect(0, 0, 0, 0)

            // calculate the right font size
            pen.style = Paint.Style.FILL_AND_STROKE
            pen.color = Color.YELLOW
            pen.strokeWidth = 2F

            pen.textSize = MAX_FONT_SIZE
            pen.getTextBounds(it.text, 0, it.text.length, tagSize)
            val fontSize: Float = pen.textSize * box.width() / tagSize.width()

            // adjust the font size so texts are inside the bounding box
            if (fontSize < pen.textSize) pen.textSize = fontSize

            var margin = (box.width() - tagSize.width()) / 2.0F
            if (margin < 0F) margin = 0F
            canvas.drawText(
                it.text, box.left + margin,
                box.top + tagSize.height().times(1F), pen
            )
        }
        return outputBitmap
    }
}