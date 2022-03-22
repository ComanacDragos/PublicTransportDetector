package ro.ubb.mobile_app.ocr

import android.graphics.Bitmap
import ro.ubb.mobile_app.core.toBase64
import ro.ubb.mobile_app.ocr.ocrSpaceApi.OcrSpaceEngineHandler
import ro.ubb.mobile_app.ocr.ocrSpaceApi.Response

object OCR {
    suspend fun detect(base64: String): Response?{
        val secondHandler = OcrSpaceEngineHandler(null, OcrSpaceEngineHandler.ENGINE.ENGINE2)
        val firstHandler = OcrSpaceEngineHandler(secondHandler, OcrSpaceEngineHandler.ENGINE.ENGINE1)
        return firstHandler.handle(base64)
    }

    suspend fun detect(bitmap: Bitmap): Response?{
        return detect(toBase64(bitmap))
    }
}