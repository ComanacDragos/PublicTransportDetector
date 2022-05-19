package ro.ubb.mobile_app.image.viewModel

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import java.util.*

class ImageFragmentViewModel(application: Application) : AndroidViewModel(application) {
    /**
     * Needed when the phone rotates, or the page is changed
     */
    lateinit var currentPhotoPath: String
    var bitmap: Bitmap? = null
}