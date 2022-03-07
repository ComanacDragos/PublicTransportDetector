package ro.ubb.mobile_app.live

import android.graphics.*
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.detection.DetectionResult

class OverlaySurfaceView(surfaceView: SurfaceView) :
    SurfaceView(surfaceView.context), SurfaceHolder.Callback {

    init {
        surfaceView.holder.addCallback(this)
        surfaceView.setZOrderOnTop(true)
    }

    private var surfaceHolder = surfaceView.holder
    private val paint = Paint()

    override fun surfaceCreated(holder: SurfaceHolder) {
        //Make surfaceView transparent
        surfaceHolder.setFormat(PixelFormat.TRANSPARENT)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
    }

    fun draw(detectedObjectList: List<DetectionResult>){
        val canvas: Canvas? = surfaceHolder.lockCanvas()
        canvas?.drawColor(0, PorterDuff.Mode.CLEAR)
        detectedObjectList.map{
            detectionObject ->
            paint.apply {
                color = when(detectionObject.classIndex){
                    0-> Color.RED
                    1-> Color.GREEN
                    2-> Color.BLUE
                    else -> Color.RED
                }
                style = Paint.Style.STROKE
                strokeWidth = 7f
                isAntiAlias = false
            }
            canvas?.drawRect(detectionObject.boundingBox, paint)

            paint.apply {
                style = Paint.Style.FILL
                isAntiAlias = true
                textSize = 77f
            }
            canvas?.drawText(
                detectionObject.label + " " + "%,.2f".format(detectionObject.score * 100) + "%",
                detectionObject.boundingBox.left,
                detectionObject.boundingBox.top - 5f,
                paint
            )
        }
        surfaceHolder.unlockCanvasAndPost(canvas ?: return)
    }
}