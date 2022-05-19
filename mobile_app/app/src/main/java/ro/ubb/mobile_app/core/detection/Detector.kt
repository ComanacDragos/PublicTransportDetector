package ro.ubb.mobile_app.core.detection

import android.graphics.*
import android.util.Log
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import ro.ubb.mobile_app.MainActivity
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.core.configuration.Configuration
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

    /**
     * Sets the new configuration
     * If modelName, maxNoBoxes or the scoreThreshold variables are changed,
     * then the detector is initialized again with the new parameters,
     * otherwise, only the configuration is changed
     * @param configuration new configuration
     */
    fun setConfiguration(configuration: Configuration){
        Log.v(TAG, "Detector settings:\n" +
                "name: ${configuration.modelName}\n" +
                "maxBoxes: ${configuration.maxNoBoxes}\n" +
                "minScore: ${configuration.scoreThreshold}\n" +
                "nmsIOU: ${configuration.nmsIouThreshold}\n")

        if(!this::configuration.isInitialized
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
                MainActivity.applicationContext(),
                configuration.modelName,
                options
            )

        }
        this.configuration = configuration
    }

    /**
     * Performs the object detection pipeline on a Bitmap
     * 1. The bitmap is converted to a TensorImage
     * 2. The image is run through the object detector
     * 3. The results are converted to [DetectionResult]
     * 4. Non-maximum Suppression (NMS) is applied in order to filter extra boxes
     * @param bitmap the input bitmap
     * @return results of object detection, as a list
     */
    private fun runDetection(bitmap: Bitmap): List<DetectionResult> {
        var start = System.currentTimeMillis()
        val image = TensorImage.fromBitmap(bitmap)
        val results = detector.detect(image)
            .map {
            val category = it.categories.first()
            val label = category.label
            val score = category.score
            DetectionResult(RectF().apply {
                top = it.boundingBox.top
                left = it.boundingBox.left
                bottom = it.boundingBox.right
                right = it.boundingBox.bottom
            }, label, score, category.index)
        }
            .filter { it.boundingBox.width() != 0f && it.boundingBox.height() != 0f }

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

    /**
     * Performs Intersection over Union (IOU)
     * @param box bounding box
     * @param otherBox the other bounding box
     * @return the division between area of the intersection and are of union between the 2 boxes
     */
    fun intersectionOverUnion(box: RectF, otherBox: RectF): Float{
        val intersectWidth = max(min(box.right, otherBox.right) - max(box.left, otherBox.left), 0f)
        val intersectHeight = max(min(box.bottom, otherBox.bottom) - max(box.top, otherBox.top), 0f)

        val intersect = intersectHeight * intersectWidth
        val union = box.height() * box.width() + otherBox.height() * otherBox.width() - intersect
        return intersect / union
    }

    /**
     * Filters out extra boxes that overlap too much (high IOU) with other boxes with high scores (NMS)
     * @param results list of unfiltered object detection results
     * @return the input list of results, but filtered
     */
    fun nonMaximumSuppression(results: List<DetectionResult>): List<DetectionResult>{
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

    /**
     * Calls the runDetection method, but firstly the bitmap is rescaled
     * @param bitmap the initial bitmap
     * @return list of object detection results on the rescaled bitmap
     */
    fun detect(bitmap: Bitmap): List<DetectionResult>{
        return runDetection(Bitmap.createScaledBitmap(
            bitmap,
            IMAGE_WIDTH, IMAGE_HEIGHT, false
        ))
    }

    /**
     * Performs object detection on a bitmap and returns a copy that has bounding boxes drawn on it
     * @param bitmap the initial bitmap
     * @return another bitmap, but with the prediction drawn over it
     */
    fun imageWithBoxes(bitmap: Bitmap): Bitmap{
        val resultToDisplay = detect(bitmap)
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)
        drawDetectionResult(canvas, resultToDisplay)
        return outputBitmap
    }

    /**
     * Logs a list of object detection results
     * @param results list of object detection results
     */
    private fun logResults(results : List<DetectionResult>) {
        Log.d(TAG, "#detections: ${results.size}")
        for ((i, obj) in results.withIndex()) {
            val box = obj.boundingBox
            Log.d(TAG, "Detected object: $i, w: ${box.width()} h: ${box.height()} ${box.width().toDouble() != 0.0 && box.height().toDouble() != 0.0}")
            Log.d(TAG, "  boundingBox: (${box.left}, ${box.top}) - (${box.right},${box.bottom})")
            Log.d(TAG, "    Label: ${obj.label}")
            Log.d(TAG, "    Confidence: ${obj.score.times(100).toInt()}%")
        }
    }

    /**
     * Merges two lists of object detection results, then applies NMS on the merged list
     * @param detections list of object detection results
     * @param newDetections list of object detection results
     * @return merged and filtered list of object detection results
     */
    fun mergeDetections(detections: List<DetectionResult>, newDetections: List<DetectionResult>): List<DetectionResult>{
        val finalResults: MutableList<DetectionResult> = LinkedList(detections)
        finalResults.addAll(newDetections)
        return nonMaximumSuppression(finalResults)
    }

    /**
     * Draws the given object detection results on the given canvas
     * @param canvas can be a static image, or the live detection screen
     * @param detectionResults object detection predictions
     */
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
                textSize = canvas.height/20f
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