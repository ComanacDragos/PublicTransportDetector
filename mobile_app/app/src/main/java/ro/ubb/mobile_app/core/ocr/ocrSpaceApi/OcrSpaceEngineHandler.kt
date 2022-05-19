package ro.ubb.mobile_app.core.ocr.ocrSpaceApi

import android.util.Log
import okhttp3.MultipartBody
import ro.ubb.mobile_app.MainActivity
import ro.ubb.mobile_app.R
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.core.ocr.core.BaseHandler
import ro.ubb.mobile_app.core.ocr.core.Handler

class OcrSpaceEngineHandler(next: Handler?, private val engine: ENGINE): BaseHandler(next){
    /**
     * Specific handler for [OCR space](https://ocr.space/OCRAPI)
     * If there is an error, or the called engine does not detect any text, the next handler is called
     * @see BaseHandler
     */
    override suspend fun handle(base64: String): Response? {
        val engineRepresentation = when(engine){
            ENGINE.ENGINE1 -> "1"
            ENGINE.ENGINE2 -> "2"
        }
        Log.v(TAG, "Handle for engine $engineRepresentation")
        try {
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("apikey", MainActivity.applicationContext().getString(R.string.ocr_space_key))
                .addFormDataPart("OCREngine", engineRepresentation)
                .addFormDataPart("base64Image", base64
                )
                .build()
            val response = OcrApi.service.detect(requestBody)
            Log.v(TAG, "Received response: $response")
            if(response.IsErroredOnProcessing){
                Log.e(TAG, "Error on api call with code: ${response.OCRExitCode}")
                return super.handle(base64)
            }
            if(response.getParsedText().trim().isEmpty()){
                Log.v(TAG, "Empty results")
                return super.handle(base64)
            }
            return response
        }catch (ex: Exception){
            Log.e(TAG, "ERROR:\n${ex.stackTraceToString()}")
            return super.handle(base64)
        }
    }

    enum class ENGINE{
        ENGINE1,
        ENGINE2
    }
}