package ro.ubb.mobile_app.image.viewModel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import java.util.*

class ImageFragmentViewModel(application: Application) : AndroidViewModel(application) {
    lateinit var currentPhotoPath: String
    var base64 = ""
}