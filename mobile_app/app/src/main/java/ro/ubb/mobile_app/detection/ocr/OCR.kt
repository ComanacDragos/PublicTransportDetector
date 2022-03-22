package ro.ubb.mobile_app.detection.ocr

import android.graphics.Bitmap
import android.util.Base64
import ro.ubb.mobile_app.detection.ocr.ocrSpaceApi.OcrSpaceEngineHandler
import ro.ubb.mobile_app.detection.ocr.ocrSpaceApi.Response
import java.io.ByteArrayOutputStream

object OCR {
    suspend fun detect(base64: String): Response?{
        val secondHandler = OcrSpaceEngineHandler(null, OcrSpaceEngineHandler.ENGINE.ENGINE2)
        val firstHandler = OcrSpaceEngineHandler(secondHandler, OcrSpaceEngineHandler.ENGINE.ENGINE1)
        return firstHandler.handle(base64)
    }

    suspend fun detect(bitmap: Bitmap): Response?{
        return detect(toBase64(bitmap))
    }

    fun toBase64(bitmap: Bitmap): String{
        val byteArrayOutputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 10, byteArrayOutputStream)
        val byteArray: ByteArray = byteArrayOutputStream.toByteArray()
        val encoded = "data:image/jpeg;base64," + Base64.encodeToString(byteArray, Base64.DEFAULT)
        return encoded.replace("\n", "")
    }
}