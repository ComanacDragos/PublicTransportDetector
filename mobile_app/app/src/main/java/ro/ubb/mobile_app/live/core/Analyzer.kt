package ro.ubb.mobile_app.live.core

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.media.Image
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.detection.DetectionResult
import ro.ubb.mobile_app.detection.Detector
import ro.ubb.mobile_app.live.yuv.YuvToRgbConverter
import kotlin.system.measureTimeMillis

typealias AnalyzerCallback = (image: List<DetectionResult>) -> Unit
class Analyzer (
    private val yuvToRgbConverter: YuvToRgbConverter,
    private val listener: AnalyzerCallback
) : ImageAnalysis.Analyzer {

    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(image: ImageProxy) {
        if(image.image == null) return
        val elapsedTime = measureTimeMillis {
            listener(detect(image.image!!))
        }
        Log.v(TAG, "Total time: ${elapsedTime}ms FPS: ${1000f/elapsedTime}")
        image.close()
    }

    private fun detect(targetImage: Image): List<DetectionResult> {
        Log.v(TAG, "targetImage width: ${targetImage.width} height: ${targetImage.height}")
        val targetBitmap = Bitmap.createBitmap(targetImage.width, targetImage.height, Bitmap.Config.ARGB_8888)
        yuvToRgbConverter.yuvToRgb(targetImage, targetBitmap)
        return Detector.detect(targetBitmap)
    }
}