package ro.ubb.mobile_app.detection

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.detection.configuration.Configuration
import ro.ubb.mobile_app.detection.configuration.local.ConfigurationDatabase
import ro.ubb.mobile_app.detection.configuration.local.ConfigurationRepository

class DetectionViewModel(application: Application) : AndroidViewModel(application) {
    private val mutableLoading = MutableLiveData<Boolean>().apply { value = false }
    private val mutableBitmap = MutableLiveData<Bitmap>().apply { value = null }
    private val mutableError = MutableLiveData<Exception>().apply { value = null }

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

    fun initDetector(context: Context, configuration: Configuration){
        if(!Detector.isDetectorInitialized())
            Detector.setConfiguration(context, configuration)
    }

    suspend fun setConfiguration(context: Context, configuration: Configuration) {
        Log.v(TAG, "Setting configuration: $configuration")
        mutableLoading.postValue(true)
        mutableError.postValue(null)
        try{
            Detector.setConfiguration(context, configuration)
            configurationRepository.setConfiguration(configuration)
        }catch (error: Exception){
            mutableError.postValue(error)
            Log.v(TAG, "ERROR:\n${error.stackTraceToString()}")
        }
        mutableLoading.postValue(false)
    }

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
}