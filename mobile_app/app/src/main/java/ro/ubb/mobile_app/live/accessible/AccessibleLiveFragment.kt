package ro.ubb.mobile_app.live.accessible

import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.Preview
import androidx.lifecycle.ViewModelProvider
import kotlinx.android.synthetic.main.fragment_live_accessible.*
import ro.ubb.mobile_app.R
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.detection.DetectionResult
import ro.ubb.mobile_app.live.AbstractLiveFragment
import java.util.*
import kotlin.collections.HashMap


class AccessibleLiveFragment: AbstractLiveFragment(), TextToSpeech.OnInitListener {
    private lateinit var tts: TextToSpeech
    private lateinit var accessibleViewModel: AccessibleLiveFragmentViewModel

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
    }

    private fun speakOut(text: String) {
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null,"")
    }

    override fun listener(detectedObjectList: List<DetectionResult>) {
        accessibleViewModel.mergeDetections(detectedObjectList)
    }

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
        super.setMenuVisibility(menuVisible)
        if(!menuVisible) {
            stopTalking()
        }else{
            tts = TextToSpeech(requireContext(), this)
            onInit(TextToSpeech.SUCCESS)
            voiceView.tts = tts
        }
    }

    override fun getPreview(): Preview? {
        //Returns null in order to not show the image preview
        return null
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_live_accessible, container, false)
    }
}