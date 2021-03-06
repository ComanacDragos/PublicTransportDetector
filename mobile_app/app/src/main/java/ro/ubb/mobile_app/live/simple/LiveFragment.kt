package ro.ubb.mobile_app.live.simple

import android.graphics.Bitmap
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.Preview
import kotlinx.android.synthetic.main.fragment_live.*
import ro.ubb.mobile_app.R
import ro.ubb.mobile_app.core.detection.DetectionResult
import ro.ubb.mobile_app.core.detection.Detector
import ro.ubb.mobile_app.live.core.AbstractLiveFragment


class LiveFragment : AbstractLiveFragment() {
    private lateinit var detectionView: DetectionView

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        detectionView = DetectionView(resultView)
    }

    /**
     * Performs object detection on the given bitmap and calls [DetectionView.draw] in order to draw
     * them on the screen
     * @param bitmap input bitmap
     */
    override fun listener(bitmap: Bitmap) {
        detectionView.draw(Detector.detect(bitmap))
    }

    /**
     * @return Returns the preview on which the live images from the camera are drawn
     */
    override fun getPreview(): Preview {
        return Preview.Builder().build()
            .also { it.setSurfaceProvider(cameraView.surfaceProvider) }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_live, container, false)
    }
}