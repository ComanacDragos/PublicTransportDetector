package ro.ubb.mobile_app.image

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.android.synthetic.main.fragment_image.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import ro.ubb.mobile_app.BuildConfig
import ro.ubb.mobile_app.R
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.detection.DetectionViewModel
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import kotlin.system.measureTimeMillis
import android.media.ExifInterface
import android.widget.Toast
import ro.ubb.mobile_app.core.toBase64
import ro.ubb.mobile_app.detection.configuration.ConfigDialog

class ImageFragment : Fragment() {
    companion object{
        private const val REQUEST_PERMISSION = 10
        private const val REQUEST_IMAGE_CAPTURE = 1
        private const val REQUEST_PICK_IMAGE = 2
    }

    private lateinit var detectionViewModel: DetectionViewModel
    private lateinit var imageFragmentViewModel: ImageFragmentViewModel

    override fun onResume() {
        super.onResume()
        checkCameraPermission()
    }

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(requireActivity(),
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_PERMISSION)
        }
    }

    private fun openCamera() {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { intent ->
            intent.resolveActivity(requireContext().packageManager)?.also {
                val photoFile: File? = try {
                    createCapturedPhoto()
                } catch (ex: IOException) {
                    Log.v(TAG, "ERROR: ${ex.stackTrace}")
                    null
                }
                Log.d(TAG, "photo file $photoFile")
                photoFile?.also {
                    val photoURI = FileProvider.getUriForFile(
                        requireContext(),
                        BuildConfig.APPLICATION_ID + ".fileprovider",
                        it
                    )
                    intent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    startActivityForResult(intent, REQUEST_IMAGE_CAPTURE)
                }
            }
        }
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.type = "image/*"
        startActivityForResult(intent, REQUEST_PICK_IMAGE)
    }

    private fun rotateImage(source: Bitmap, angle: Float): Bitmap {
        Log.v(TAG, "Rotate with angle: $angle")
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height,
            matrix, true
        )
    }

    private fun rotateIfPossible(bitmap: Bitmap, uri: Uri): Bitmap{
        try{
            val ei = ExifInterface(uri.path!!)
            val orientation: Int = ei.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_UNDEFINED
            )

            return when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> rotateImage(bitmap, 90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> rotateImage(bitmap, 180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> rotateImage(bitmap, 270f)
                ExifInterface.ORIENTATION_NORMAL -> bitmap
                else -> bitmap
            }
        }catch (ex: Exception){
            Log.v(TAG, "COULD NOT ROTATE: ${ex.message}")
            return bitmap
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        try{
            if (resultCode == AppCompatActivity.RESULT_OK) {
                var uri: Uri? = null
                if (requestCode == REQUEST_IMAGE_CAPTURE) {
                    uri = Uri.fromFile(File(imageFragmentViewModel.currentPhotoPath))
                }
                else if (requestCode == REQUEST_PICK_IMAGE) {
                    uri = data?.data
                }

                uri?.let {
                    requireContext().contentResolver.openInputStream(it)
                }.also {
                    var bitmap = BitmapFactory.decodeStream(it)
                    bitmap = rotateIfPossible(bitmap, uri!!)
                    lifecycleScope.launch(Dispatchers.Default) {
                        imageFragmentViewModel.base64 = toBase64(bitmap)
                        Log.v(TAG, "width: ${bitmap.width} height: ${bitmap.height} starting detection...")

                        val elapsed = measureTimeMillis {
                            detectionViewModel.detect(bitmap)
                        }
                        Log.v(TAG, "Total detection time: ${elapsed}ms")
                    }

                    requireActivity().runOnUiThread{
                        ivImage.setImageBitmap(bitmap)
                    }
                    it!!.close()
                }
            }
        }catch (ex: Exception){
            Log.v(TAG, "ERROR: ${ex.stackTraceToString()}")
        }
    }

    @Throws(IOException::class)
    private fun createCapturedPhoto(): File {
        val timestamp: String = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val storageDir = requireContext().getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("PHOTO_${timestamp}",".jpg", storageDir).apply {
            imageFragmentViewModel.currentPhotoPath = absolutePath
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_image, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        captureImage.setOnClickListener { openCamera() }
        openGalleryButton.setOnClickListener { openGallery() }
        detectionViewModel = ViewModelProvider(this)[DetectionViewModel::class.java]
        imageFragmentViewModel = ViewModelProvider(this)[ImageFragmentViewModel::class.java]

        detectionViewModel.loading.observe(viewLifecycleOwner, {
            Log.v(TAG, "update detecting")
            progressBar.visibility = if (it) View.VISIBLE else View.GONE
            captureImage.isEnabled = !it
            openGalleryButton.isEnabled = !it
            ocrButton.isEnabled = !it
        })

        detectionViewModel.bitmap.observe(viewLifecycleOwner, {
            Log.v(TAG, "update bitmap")
            requireActivity().runOnUiThread{
                ivImage.setImageBitmap(it)
            }
        })

        detectionViewModel.error.observe(viewLifecycleOwner, {
            if(it != null)
                errorTextView.text = it.message
            else
                errorTextView.text = ""
        })

        detectionViewModel.configuration.observe(viewLifecycleOwner, {
            if(it!=null){
                detectionViewModel.initDetector(it)
            }
        })

        settingsFab.setOnClickListener {
            ConfigDialog().show(requireActivity().supportFragmentManager, "ConfigDialog")
        }

        detectionViewModel.ocrString.observe(viewLifecycleOwner, {
            ocrTextView.text = it
        })

        ocrButton.setOnClickListener {
            if(imageFragmentViewModel.base64 == ""){
                Toast.makeText(requireContext(),
                    "No image to perform OCR on",
                    Toast.LENGTH_LONG).show()
            }else{
                Toast.makeText(requireContext(),
                    "Starting OCR...",
                    Toast.LENGTH_LONG).show()
                lifecycleScope.launch(Dispatchers.Default) {
                    detectionViewModel.ocr(imageFragmentViewModel.base64)
                }
            }
        }
    }
}