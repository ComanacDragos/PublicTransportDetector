package ro.ubb.mobile_app.live.accessible

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.core.detection.DetectionResult
import ro.ubb.mobile_app.core.detection.Detector
import ro.ubb.mobile_app.core.ocr.OCR
import ro.ubb.mobile_app.core.toBase64
import java.lang.Exception
import java.util.*

class AccessibleLiveFragmentViewModel(application: Application) : AndroidViewModel(application) {
    private val mutableDetections = MutableLiveData<List<DetectionResult>>().apply { value = LinkedList() }
    val detections: LiveData<List<DetectionResult>> = mutableDetections

    private val mutableOcrString = MutableLiveData<String>().apply {value = ""}
    val ocrString: LiveData<String> = mutableOcrString

    var ocrInProgress = false

    /**
     * Calls the [OCR.detect] on a given bitmap
     * @param bitmap input bitmap
     */
    suspend fun ocr(bitmap: Bitmap){
        ocrInProgress = true
        try{
            OCR.detect(bitmap)?.apply { mutableOcrString.postValue(this.getParsedText()) }
        }catch (ex: Exception){
            Log.e(TAG, ex.stackTraceToString())
        }
        ocrInProgress = false
    }

    /**
     * Merges the new detections with the current detections using [Detector.mergeDetections]
     * @param newDetections the new list of predicted bounding boxes
     */
    fun mergeDetections(newDetections: List<DetectionResult>){
        mutableDetections.postValue(mutableDetections.value?.let
            { Detector.mergeDetections(it, newDetections) })
    }

    /**
     * Empties the list of bounding boxes
     */
    fun resetDetections(){
        mutableDetections.postValue(LinkedList())
    }
}