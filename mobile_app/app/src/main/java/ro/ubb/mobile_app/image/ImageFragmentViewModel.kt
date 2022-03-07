package ro.ubb.mobile_app.image

import android.app.Application
import androidx.lifecycle.AndroidViewModel

class ImageFragmentViewModel(application: Application) : AndroidViewModel(application) {
    lateinit var currentPhotoPath: String
}