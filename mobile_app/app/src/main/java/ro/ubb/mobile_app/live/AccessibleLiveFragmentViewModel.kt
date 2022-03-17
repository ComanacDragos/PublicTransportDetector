package ro.ubb.mobile_app.live

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import ro.ubb.mobile_app.detection.DetectionResult
import ro.ubb.mobile_app.detection.Detector
import java.util.*

class AccessibleLiveFragmentViewModel(application: Application) : AndroidViewModel(application) {
    private val mutableDetections = MutableLiveData<List<DetectionResult>>().apply { value = LinkedList() }
    val detections: LiveData<List<DetectionResult>> = mutableDetections

    fun mergeDetections(newDetections: List<DetectionResult>){
        val mergedDetections =
            mutableDetections.value?.let { Detector.mergeDetections(it, newDetections) }
        if(mergedDetections != null)
            for (detection in mergedDetections){
                if(!mutableDetections.value!!.contains(detection)) {
                    mutableDetections.postValue(mergedDetections)
                    break
                }
            }
    }

    fun resetDetections(){
        mutableDetections.postValue(LinkedList())
    }
}