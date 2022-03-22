package ro.ubb.mobile_app.ocr.ocrSpaceApi

import okhttp3.RequestBody
import retrofit2.http.Body
import retrofit2.http.POST


object OcrApi {
    interface Service{
        @POST("/parse/image")
        suspend fun detect(@Body body: RequestBody): Response
    }
    val service: Service = Api.retrofit.create(Service::class.java)
}