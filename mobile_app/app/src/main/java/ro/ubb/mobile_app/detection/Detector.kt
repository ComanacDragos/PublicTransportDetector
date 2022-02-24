package ro.ubb.mobile_app.detection

import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import ro.ubb.mobile_app.core.TAG

class Detector(val context: Context, val modelName: String) {
    companion object {
        private const val MAX_FONT_SIZE = 96F
    }

    private var detector: ObjectDetector

    init {
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setMaxResults(5)
            .setScoreThreshold(0.5f)
            .build()
        detector = ObjectDetector.createFromFileAndOptions(
            context,
            modelName,
            options
        )
    }

    private fun runDetection(bitmap: Bitmap): MutableList<Detection>? {
        val image = TensorImage.fromBitmap(bitmap)
        val results = detector.detect(image)
        debugPrint(results)
        return results
    }

    fun imageWithBoxes(bitmap: Bitmap): Bitmap{
        val results = runDetection(bitmap)
        val resultToDisplay = results!!.map {
            val category = it.categories.first()
            val text = "${category.label}, ${category.score.times(100).toInt()}%"
            DetectionResult(it.boundingBox, text)
        }
        return drawDetectionResult(bitmap, resultToDisplay)
    }

    private fun debugPrint(results : List<Detection>) {
        Log.v(TAG, "#detections: ${results.size}")
        for ((i, obj) in results.withIndex()) {
            val box = obj.boundingBox

            Log.v(TAG, "Detected object: ${i} ")
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
            pen.color = Color.RED
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