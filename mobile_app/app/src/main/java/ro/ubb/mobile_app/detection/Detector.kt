package ro.ubb.mobile_app.detection

import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.detection.configuration.Configuration
import java.util.*
import kotlin.math.max
import kotlin.math.min

object Detector{
    private const val IMAGE_WIDTH = 416
    private const val IMAGE_HEIGHT = 416

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

    private fun runDetection(bitmap: Bitmap): List<DetectionResult> {
        var start = System.currentTimeMillis()
        val image = TensorImage.fromBitmap(bitmap)
        val results = detector.detect(image).map {
            val category = it.categories.first()
            val label = category.label
            val score = category.score
            DetectionResult(it.boundingBox, label, score, category.index)
        }
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

    private fun nonMaximumSuppression(results: List<DetectionResult>): List<DetectionResult>{
        val newResults = LinkedList<DetectionResult>()
        results.sortedByDescending { it.score }
            .forEach{
                var maxIou = -1f
                for(result in newResults){
                    if(result.label == it.label){
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
        return runDetection(Bitmap.createScaledBitmap(
            bitmap,
            IMAGE_WIDTH, IMAGE_HEIGHT, false
        ))
    }

    fun imageWithBoxes(bitmap: Bitmap): Bitmap{
        val resultToDisplay = detect(bitmap)
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)
        drawDetectionResult(canvas, resultToDisplay)
        return outputBitmap
    }

    private fun logResults(results : List<DetectionResult>) {
        Log.d(TAG, "#detections: ${results.size}")
        for ((i, obj) in results.withIndex()) {
            val box = obj.boundingBox
            Log.d(TAG, "Detected object: $i ")
            Log.d(TAG, "  boundingBox: (${box.left}, ${box.top}) - (${box.right},${box.bottom})")
            Log.d(TAG, "    Label: ${obj.label}")
            Log.d(TAG, "    Confidence: ${obj.score.times(100).toInt()}%")
        }
    }

    fun mergeDetections(detections: List<DetectionResult>, newDetections: List<DetectionResult>): List<DetectionResult>{
        val finalResults: MutableList<DetectionResult> = LinkedList(detections)
        finalResults.addAll(newDetections)
        return nonMaximumSuppression(finalResults)
    }

    fun drawDetectionResult(
        canvas: Canvas,
        detectionResults: List<DetectionResult>
    ) {
        val paint = Paint()
        detectionResults.map{
                detectionObject ->
            paint.apply {
                color = when(detectionObject.classIndex){
                    0-> Color.RED
                    1-> Color.GREEN
                    2-> Color.BLUE
                    else -> Color.MAGENTA
                }
                style = Paint.Style.STROKE
                strokeWidth = 7f
                isAntiAlias = false
            }

            val boundingBox = RectF().apply {
                top=detectionObject.boundingBox.top/ IMAGE_HEIGHT*canvas.height
                left=detectionObject.boundingBox.left/ IMAGE_WIDTH*canvas.width
                bottom=detectionObject.boundingBox.bottom/IMAGE_HEIGHT*canvas.height
                right=detectionObject.boundingBox.right/IMAGE_WIDTH*canvas.width
            }
            canvas.drawRect(boundingBox, paint)

            paint.apply {
                style = Paint.Style.FILL
                isAntiAlias = true
                textSize = 36f
            }
            canvas.drawText(
                detectionObject.label + " " + "%,.2f".format(detectionObject.score * 100) + "%",
                boundingBox.left,
                boundingBox.top - 5f,
                paint
            )
        }
    }
}