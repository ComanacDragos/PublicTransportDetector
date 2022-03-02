package ro.ubb.mobile_app.image

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
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
import ro.ubb.mobile_app.detection.DetectionViewModel
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

import ro.ubb.mobile_app.core.TAG
import kotlin.system.measureTimeMillis


class ImageFragment : Fragment() {
    companion object{
        private val REQUEST_PERMISSION = 10
        private val REQUEST_IMAGE_CAPTURE = 1
        private val REQUEST_PICK_IMAGE = 2
    }

    private lateinit var currentPhotoPath: String
    private lateinit var detectionViewModel: DetectionViewModel

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
                    null
                }
                Log.d("MainActivity", "photofile $photoFile");
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

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == AppCompatActivity.RESULT_OK) {
            var uri: Uri? = null
            if (requestCode == REQUEST_IMAGE_CAPTURE) {
                uri = Uri.fromFile(File(currentPhotoPath))
            }
            else if (requestCode == REQUEST_PICK_IMAGE) {
                uri = data?.data
            }
            //ivImage.setImageURI(uri)
            //return
            uri?.let {
                requireContext().contentResolver.openInputStream(it)
            }.also {
//                val bitmap = Bitmap.createScaledBitmap(
//                    BitmapFactory.decodeStream(it),
//                    416, 416, false
//                )

                val bitmap = BitmapFactory.decodeStream(it)

                lifecycleScope.launch(Dispatchers.Default) {
                    Log.v(TAG, "start detection")
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
    }

    @Throws(IOException::class)
    private fun createCapturedPhoto(): File {
        val timestamp: String = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val storageDir = requireContext().getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("PHOTO_${timestamp}",".jpg", storageDir).apply {
            currentPhotoPath = absolutePath
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_image, container, false)
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        captureVideo.setOnClickListener { openCamera() }
        openGalleryButton.setOnClickListener { openGallery() }
        detectionViewModel = ViewModelProvider(this)[DetectionViewModel::class.java]

        detectionViewModel.loading.observe(viewLifecycleOwner, {
            Log.v(TAG, "update detecting")
            progressBar.visibility = if (it) View.VISIBLE else View.GONE
            captureVideo.isEnabled = !it
            openGalleryButton.isEnabled = !it
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
        })

        detectionViewModel.configuration.observe(viewLifecycleOwner, {
            errorTextView.text = "$it"
        })

        detectionViewModel.configuration.observe(viewLifecycleOwner, {
            if(it!=null){
                detectionViewModel.initDetector(requireContext(), it)
            }
        })
    }
}