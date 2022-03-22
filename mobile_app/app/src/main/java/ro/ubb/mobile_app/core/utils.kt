package ro.ubb.mobile_app.core

import android.graphics.Bitmap
import android.util.Base64
import java.io.ByteArrayOutputStream

fun toBase64(bitmap: Bitmap): String{
    val byteArrayOutputStream = ByteArrayOutputStream()
    bitmap.compress(Bitmap.CompressFormat.JPEG, 10, byteArrayOutputStream)
    val byteArray: ByteArray = byteArrayOutputStream.toByteArray()
    val encoded = "data:image/jpeg;base64," + Base64.encodeToString(byteArray, Base64.DEFAULT)
    return encoded.replace("\n", "")
}