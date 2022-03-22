package ro.ubb.mobile_app.live.core

import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.detection.DetectionResult
import ro.ubb.mobile_app.live.yuv.YuvToRgbConverter
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

abstract class AbstractLiveFragment : Fragment(){
    private var cameraIsSetup = false
    private lateinit var cameraExecutor: ExecutorService
    private val yuvToRgbConverter: YuvToRgbConverter by lazy {
        YuvToRgbConverter(requireContext())
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    protected abstract fun listener(detectedObjectList: List<DetectionResult>)
    protected abstract fun getPreview(): Preview?


    private fun setupCamera(){
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(5000, 5000)) // large numbers so that the highest resolution is picked
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(
                        cameraExecutor,
                        Analyzer(
                            yuvToRgbConverter
                        ){
                                detectedObjectList -> listener(detectedObjectList)
                        }
                    )
                }

            try {
                cameraProvider.unbindAll()

                val preview = getPreview()
                if(preview != null)
                    cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
                else
                    cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalyzer)
                cameraIsSetup = true
            } catch (ex: Exception) {
                Log.e(TAG, "ERROR: Camera Use case binding failed", ex)
            }

        }, ContextCompat.getMainExecutor(requireContext()))
    }

    override fun onDestroy() {
        super.onDestroy()
        if(this::cameraExecutor.isInitialized)
            cameraExecutor.shutdown()
    }

    override fun setMenuVisibility(menuVisible: Boolean) {
        super.setMenuVisibility(menuVisible)
        if(menuVisible && !cameraIsSetup)
            setupCamera()
        else{
            if(!cameraIsSetup)
                return
            val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
            cameraProviderFuture.addListener({
                val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
                try{
                    cameraProvider.unbindAll()
                    cameraIsSetup = false
                }catch (ex: Exception){
                    Log.v(TAG, "ERROR: on closing camera: \n${ex.stackTraceToString()}")
                }
            }, ContextCompat.getMainExecutor(requireContext()))
        }
    }
}