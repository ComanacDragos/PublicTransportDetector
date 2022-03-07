package ro.ubb.mobile_app.live

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import kotlinx.android.synthetic.main.fragment_live.*
import ro.ubb.mobile_app.R
import ro.ubb.mobile_app.core.TAG
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class LiveFragment : Fragment() {
    private var fragmentIsChosen: Boolean = false
    private lateinit var cameraExecutor: ExecutorService
    private val yuvToRgbConverter: YuvToRgbConverter by lazy {
        YuvToRgbConverter(requireContext())
    }

    private lateinit var overlaySurfaceView: OverlaySurfaceView

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        overlaySurfaceView = OverlaySurfaceView(resultView)
        cameraExecutor = Executors.newSingleThreadExecutor()
        setupCamera()
    }

    private fun setupCamera(){
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build()
                .also { it.setSurfaceProvider(cameraView.surfaceProvider) }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetRotation(cameraView.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(
                        cameraExecutor,
                        Analyzer(
                            yuvToRgbConverter
                        ){
                            detectedObjectList ->
                                overlaySurfaceView.draw(detectedObjectList)
                        }
                    )
                }

            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e("ERROR: Camera", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(requireContext()))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun setMenuVisibility(menuVisible: Boolean) {
        super.setMenuVisibility(menuVisible)
        fragmentIsChosen = menuVisible
        Log.v(TAG, "Live fragment visibility: $fragmentIsChosen")

    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_live, container, false)
    }

}