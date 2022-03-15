package ro.ubb.mobile_app.live

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.Preview
import kotlinx.android.synthetic.main.fragment_live.*
import ro.ubb.mobile_app.R
import ro.ubb.mobile_app.detection.DetectionResult


class LiveFragment : AbstractLiveFragment() {
    private lateinit var overlaySurfaceView: OverlaySurfaceView

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        overlaySurfaceView = OverlaySurfaceView(resultView)
    }

    override fun listener(detectedObjectList: List<DetectionResult>) {
        overlaySurfaceView.draw(detectedObjectList)
    }

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