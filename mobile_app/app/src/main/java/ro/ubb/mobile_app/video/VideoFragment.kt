package ro.ubb.mobile_app.video

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import ro.ubb.mobile_app.R
import android.content.Intent

import android.provider.MediaStore
import android.net.Uri

import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.fragment_video.*


class VideoFragment : Fragment() {
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == AppCompatActivity.RESULT_OK) {
            if(data != null){
                val video: Uri? = data.data
                videoView.setVideoURI(video)
                videoView.start()
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_video, container, false)
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        openGalleryButton.setOnClickListener {
            val i = Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(i, 0)
        }
    }

}