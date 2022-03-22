package ro.ubb.mobile_app.core.ocr

import android.graphics.Bitmap
import ro.ubb.mobile_app.core.toBase64
import ro.ubb.mobile_app.core.ocr.ocrSpaceApi.OcrSpaceEngineHandler
import ro.ubb.mobile_app.core.ocr.ocrSpaceApi.Response

object OCR {
    suspend fun detect(base64: String): Response?{
        val secondHandler = OcrSpaceEngineHandler(null, OcrSpaceEngineHandler.ENGINE.ENGINE2)
        val firstHandler = OcrSpaceEngineHandler(secondHandler, OcrSpaceEngineHandler.ENGINE.ENGINE1)
        return firstHandler.handle(base64)
    }

    suspend fun detect(bitmap: Bitmap): Response?{
        if(bitmap.height > 1024 || bitmap.width > 1024){
            return detect(toBase64(Bitmap.createScaledBitmap(bitmap, 1024, 1024, false)))
        }
        return detect(toBase64(bitmap))
    }
}