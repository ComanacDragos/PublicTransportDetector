package ro.ubb.mobile_app.live.accessible

import android.graphics.Bitmap
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.Preview
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.android.synthetic.main.fragment_live_accessible.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import ro.ubb.mobile_app.R
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.core.detection.DetectionResult
import ro.ubb.mobile_app.core.detection.Detector
import ro.ubb.mobile_app.live.core.AbstractLiveFragment
import java.util.*
import kotlin.collections.HashMap


class AccessibleLiveFragment: AbstractLiveFragment(), TextToSpeech.OnInitListener {
    private lateinit var tts: TextToSpeech
    private lateinit var accessibleViewModel: AccessibleLiveFragmentViewModel
    private var errorOnSetup = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        tts = TextToSpeech(requireContext(), this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e(TAG,"The Language specified is not supported!")
            }
        } else {
            Log.e(TAG, "Initialization Failed!")
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        accessibleViewModel = ViewModelProvider(this)[AccessibleLiveFragmentViewModel::class.java]
        voiceView.tts = tts
        accessibleViewModel.detections.observe(viewLifecycleOwner, {
            if(!tts.isSpeaking && it.isNotEmpty()){
                val text = detectionsToString(it)
                    try{
                        speakOut(text)
                    }catch (ex: Exception){
                        Log.e(TAG, "ERROR:\n${ex.stackTraceToString()}")
                    }
                accessibleViewModel.resetDetections()
            }
        })

        accessibleViewModel.ocrString.observe(viewLifecycleOwner, {
            try {
                if (it.isNotEmpty())
                    speakOut(it)
            }catch (ex: Exception){
                Log.e(TAG, "ERROR:\n${ex.stackTraceToString()}")
            }
        })
    }

    /**
     * Activates the sinusoidal wave and start the TTS
     * @param text text to be played on the speakers using TTS
     */
    private fun speakOut(text: String) {
        voiceView.invalidate()
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null,"")
    }

    /**
     * Applies object detection and OCR on the given bitmap
     * @param bitmap input bitmap
     */
    override fun listener(bitmap: Bitmap) {
        accessibleViewModel.mergeDetections(Detector.detect(bitmap))
        accessibleViewModel.detections.value?.apply {
            if(!accessibleViewModel.ocrInProgress)
                lifecycleScope.launch(Dispatchers.Default) {
                    accessibleViewModel.ocr(bitmap)
                }
        }
    }

    /**
     * Converts the list of object detection results to a string
     * @param detectedObjectList list of predicted bounding boxes
     * @return the bounding boxes in the form of a string of the following format:
     * N_1 Class_1, N_2 class_2, ..., N_m class_m
     */
    private fun detectionsToString(detectedObjectList: List<DetectionResult>): String{
        val counter = HashMap<String, Int>()
        detectedObjectList.forEach{
            val count = counter[it.label]
            if(count != null)
                counter[it.label] = count+1
            else
                counter[it.label] = 1
        }
        var text = ""
        if(counter.size != 0)
            text = counter.map {
                var newLabel = if (it.key.contains("?")) "object" else it.key.trim()
                if (it.value > 1)
                    newLabel += if (newLabel.lowercase() == "bus") "ses" else "s"
                "${it.value} $newLabel"
            }.reduce{
                    acc, string -> "$acc $string"
            }
        return text
    }

    /**
     * Stops the TTS service
     */
    private fun stopTalking(){
        if(this::tts.isInitialized){
            tts.stop()
            tts.shutdown()
        }
    }

    override fun onDestroy() {
        stopTalking()
        super.onDestroy()
    }

    override fun setMenuVisibility(menuVisible: Boolean){
        try {
            super.setMenuVisibility(menuVisible)
            if (!menuVisible) {
                stopTalking()
            } else {
                tts = TextToSpeech(requireContext(), this)
                onInit(TextToSpeech.SUCCESS)
                voiceView.tts = tts
            }
        }catch (err: Exception){
            err.let { Log.e(TAG, it.stackTraceToString()) }
            errorOnSetup = true
        }
    }

    /**
     * @return null in order to not show the image preview
     */
    override fun getPreview(): Preview? {
        return null
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        Log.d(TAG, "ON CREATE VIEW")
        if(errorOnSetup){
            setMenuVisibility(true)
            errorOnSetup = false
        }
        return inflater.inflate(R.layout.fragment_live_accessible, container, false)
    }
}