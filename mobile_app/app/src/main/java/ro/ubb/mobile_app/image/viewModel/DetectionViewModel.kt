package ro.ubb.mobile_app.image.viewModel

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.core.detection.Detector
import ro.ubb.mobile_app.core.configuration.Configuration
import ro.ubb.mobile_app.core.configuration.local.ConfigurationDatabase
import ro.ubb.mobile_app.core.configuration.local.ConfigurationRepository
import ro.ubb.mobile_app.core.ocr.OCR

class DetectionViewModel(application: Application) : AndroidViewModel(application) {
    private val mutableLoading = MutableLiveData<Boolean>().apply { value = false }
    private val mutableBitmap = MutableLiveData<Bitmap>().apply { value = null }
    private val mutableError = MutableLiveData<Exception>().apply { value = null }
    private val mutableOcrString = MutableLiveData<String>().apply {value = ""}

    val ocrString: LiveData<String> = mutableOcrString

    val error: LiveData<Exception> = mutableError

    val loading: LiveData<Boolean> = mutableLoading
    val bitmap: LiveData<Bitmap> = mutableBitmap

    private val configurationRepository: ConfigurationRepository
    val configuration: LiveData<Configuration>

    init {
        val configurationDao = ConfigurationDatabase.getDatabase(application, viewModelScope).configurationDao()
        configurationRepository = ConfigurationRepository(configurationDao)
        configuration = configurationRepository.configuration
    }

    /**
     * If the detector is not initialized, it initializes it with the given configuration
     * @param configuration initial configuration which should come from the local database at startup
     */
    fun initDetector(configuration: Configuration){
        try{
            if(!Detector.isDetectorInitialized())
                Detector.setConfiguration(configuration)
        }catch (error: Exception){
            mutableError.postValue(error)
            Log.v(TAG, "ERROR:\n${error.stackTraceToString()}")
        }
    }

    /**
     * Sets a new configuration for the detector, and updates the local database
     * @param configuration new configuration
     */
    suspend fun setConfiguration(configuration: Configuration) {
        Log.v(TAG, "Setting configuration: $configuration")
        mutableLoading.postValue(true)
        mutableError.postValue(null)
        try{
            Detector.setConfiguration(configuration)
            configurationRepository.setConfiguration(configuration)
        }catch (error: Exception){
            mutableError.postValue(error)
            Log.v(TAG, "ERROR:\n${error.stackTraceToString()}")
        }
        mutableLoading.postValue(false)
    }

    /**
     * Performs object detection on the given bitmap
     * @param inputBitmap input bitmap
     */
    fun detect(inputBitmap: Bitmap){
        mutableLoading.postValue(true)
        mutableError.postValue(null)
        try{
            mutableBitmap.postValue(Detector.imageWithBoxes(inputBitmap))
            Log.v(TAG, "done detecting")
        }catch (error: Exception){
            mutableError.postValue(error)
            Log.v(TAG, "ERROR:\n${error.stackTraceToString()}")
        }
        mutableLoading.postValue(false)
    }


    /**
     * Performs OCR on the given bitmap
     * @param bitmap input bitmap
     */
    suspend fun ocr(bitmap: Bitmap){
        mutableLoading.postValue(true)
        mutableError.postValue(null)
        try{
            val response = OCR.detect(bitmap)
            if(response == null){
                mutableError.postValue(Exception("Could not perform OCR"))
            }else{
                mutableOcrString.postValue(response.getParsedText())
            }
        }catch (error: Exception){
            mutableError.postValue(error)
            Log.v(TAG, "ERROR:\n${error.stackTraceToString()}")
        }
        mutableLoading.postValue(false)
    }
}