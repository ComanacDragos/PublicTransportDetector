package ro.ubb.mobile_app.core.ocr

import android.graphics.Bitmap
import ro.ubb.mobile_app.core.toBase64
import ro.ubb.mobile_app.core.ocr.ocrSpaceApi.OcrSpaceEngineHandler
import ro.ubb.mobile_app.core.ocr.ocrSpaceApi.Response

object OCR {
    /**
     * Composes the Chain of Responsibility handlers, then calls the API
     * @param base64 image in base64
     * @return the HTTP response from OCR
     */
    suspend fun detect(base64: String): Response?{
        val secondHandler = OcrSpaceEngineHandler(null, OcrSpaceEngineHandler.ENGINE.ENGINE2)
        val firstHandler = OcrSpaceEngineHandler(secondHandler, OcrSpaceEngineHandler.ENGINE.ENGINE1)
        return firstHandler.handle(base64)
    }

    /**
     * Converts the input bitmap to base64, then calls the [OCR.detect]
     * @param bitmap input bitmap
     * @return the HTTP response from OCR
     */
    suspend fun detect(bitmap: Bitmap): Response?{
        if(bitmap.height > 1024 || bitmap.width > 1024){
            return detect(toBase64(Bitmap.createScaledBitmap(bitmap, 1024, 1024, false)))
        }
        return detect(toBase64(bitmap))
    }
}