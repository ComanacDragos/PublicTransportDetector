package ro.ubb.mobile_app.ocr.core

import ro.ubb.mobile_app.ocr.ocrSpaceApi.Response

interface Handler {
    suspend fun handle(base64: String): Response?
}

open class BaseHandler(private val next: Handler?): Handler {
    override suspend fun handle(base64: String): Response? {
        return next?.handle(base64)
    }
}