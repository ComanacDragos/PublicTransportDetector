package ro.ubb.mobile_app.live

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.media.Image
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.detection.DetectionResult
import ro.ubb.mobile_app.detection.Detector

typealias AnalyzerCallback = (image: List<DetectionResult>) -> Unit
class Analyzer (
    private val yuvToRgbConverter: YuvToRgbConverter,
    private val listener: AnalyzerCallback
) : ImageAnalysis.Analyzer {

    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(image: ImageProxy) {
        if(image.image == null) return
        listener(
            detect(image.image!!)
        )
        image.close()
    }

    //Image YUV-> RGB bitmap -> tensorflowImage ->Convert to tensorflowBuffer, infer and output the result as a list
    private fun detect(targetImage: Image): List<DetectionResult> {
        Log.v(TAG, "targetImage width: ${targetImage.width} height: ${targetImage.height}")
        val targetBitmap =
            Bitmap.createBitmap(targetImage.width, targetImage.height, Bitmap.Config.ARGB_8888)
        yuvToRgbConverter.yuvToRgb(targetImage, targetBitmap)
        return Detector.detect(Bitmap.createScaledBitmap(
            targetBitmap,
            416, 416, false
        ))
    }
}