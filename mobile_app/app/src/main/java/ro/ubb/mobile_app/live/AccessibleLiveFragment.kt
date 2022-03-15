package ro.ubb.mobile_app.live

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.Preview
import ro.ubb.mobile_app.R
import ro.ubb.mobile_app.detection.DetectionResult
import android.speech.tts.TextToSpeech
import android.util.Log
import kotlinx.android.synthetic.main.fragment_live_accessible.*
import ro.ubb.mobile_app.core.TAG
import java.util.*


class AccessibleLiveFragment: AbstractLiveFragment(), TextToSpeech.OnInitListener {
    private var tts: TextToSpeech? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        tts = TextToSpeech(requireContext(), this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            // set US English as language for tts
            val result = tts!!.setLanguage(Locale.US)

            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e(TAG,"The Language specified is not supported!")
            } else {
                buttonSpeak.isEnabled = true
            }

        } else {
            Log.e(TAG, "Initialization Failed!")
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        buttonSpeak.setOnClickListener { speakOut(edittextInput.text.toString()) }
    }

    private fun speakOut(text: String) {
        tts!!.speak(text, TextToSpeech.QUEUE_FLUSH, null,"")
    }

    override fun listener(detectedObjectList: List<DetectionResult>) {

    }

    override fun onDestroy() {
        // Shutdown TTS
        if (tts != null) {
            tts!!.stop()
            tts!!.shutdown()
        }
        super.onDestroy()
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