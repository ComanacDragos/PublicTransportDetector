package ro.ubb.mobile_app.live

import android.graphics.*
import android.view.SurfaceHolder
import android.view.SurfaceView
import ro.ubb.mobile_app.detection.DetectionResult
import ro.ubb.mobile_app.detection.Detector

class OverlaySurfaceView(surfaceView: SurfaceView) :
    SurfaceView(surfaceView.context), SurfaceHolder.Callback {

    init {
        surfaceView.holder.addCallback(this)
        surfaceView.setZOrderOnTop(true)
    }

    private var surfaceHolder = surfaceView.holder

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
        if(canvas!=null){
            canvas.drawColor(0, PorterDuff.Mode.CLEAR)
            Detector.drawDetectionResult(canvas, detectedObjectList)
       }
        surfaceHolder.unlockCanvasAndPost(canvas ?: return)
    }
}