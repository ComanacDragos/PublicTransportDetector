package ro.ubb.mobile_app.detection.configuration

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import androidx.fragment.app.DialogFragment
import androidx.lifecycle.ViewModelProvider
import kotlinx.android.synthetic.main.fragment_config.*
import ro.ubb.mobile_app.R
import android.content.res.AssetManager
import android.util.Log
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.detection.DetectionViewModel
import java.lang.Exception


class ConfigDialog: DialogFragment() {
    private lateinit var detectionViewModel: DetectionViewModel

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        dialog!!.window?.setBackgroundDrawableResource(R.drawable.round_corner)
        return inflater.inflate(R.layout.fragment_config, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        updateButton.setOnClickListener {
            lifecycleScope.launch(Dispatchers.Default) {
                Log.v(TAG, "start setting configuration")
                detectionViewModel.setConfiguration(
                    requireContext(),
                    Configuration(
                        modelSpinner.selectedItem.toString(),
                        maxNoBoxesSlider.value.toInt(),
                        minimumScoreSlider.value
                    )
                )
                dismiss()
            }
        }
        val assetManager: AssetManager = requireActivity().assets
        val files = assetManager.list("")
        if(files != null) {
            val adapter = ArrayAdapter(
                requireContext(),
                android.R.layout.simple_spinner_dropdown_item,
                files.filter {
                    it.contains("tflite")
                }
            )

            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            modelSpinner.adapter = adapter
        }
        detectionViewModel = ViewModelProvider(this)[DetectionViewModel::class.java]

        minimumScoreSlider.setLabelFormatter { "${it.toInt()}%" }

//        minimumScoreSlider.value = detectionViewModel.configuration.value!!.scoreThreshold
//        maxNoBoxesSlider.value = detectionViewModel.configuration.value!!.maxNoBoxes.toFloat()

        detectionViewModel.configuration.observe(viewLifecycleOwner, {
            if(it != null){
                minimumScoreSlider.value = it.scoreThreshold
                maxNoBoxesSlider.value = it.maxNoBoxes.toFloat()
            }
        })

    }
}

