package ro.ubb.mobile_app.detection

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import ro.ubb.mobile_app.R
import ro.ubb.mobile_app.core.TAG

class DetectionViewModel(application: Application) : AndroidViewModel(application) {
    private val mutableDetecting = MutableLiveData<Boolean>().apply { value = false }
    private val mutableBitmap = MutableLiveData<Bitmap>().apply { value = null }
    private val mutableError = MutableLiveData<Exception>().apply { value = null }

    val error: LiveData<Exception> = mutableError

    val detecting: LiveData<Boolean> = mutableDetecting
    val bitmap: LiveData<Bitmap> = mutableBitmap

     private lateinit var detector: Detector

     fun initDetector(context: Context){
        detector = Detector(context, context.resources.getString(R.string.model))
     }

    fun detect(inputBitmap: Bitmap){
        mutableDetecting.value = true
        mutableError.value = null
        try{
            mutableBitmap.value = detector.imageWithBoxes(inputBitmap)
            Log.v(TAG, "done detecting")
        }catch (error: Exception){
            mutableError.value = error
            Log.v(TAG, "ERROR: ${error.stackTrace}")
        }
        mutableDetecting.value = false
    }
}