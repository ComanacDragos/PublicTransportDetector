package ro.ubb.mobile_app.core.ocr.core

import ro.ubb.mobile_app.core.ocr.ocrSpaceApi.Response

interface Handler {
    /**
     * The Chain of Responsibility Handler interface for calling an external API
     * @param base64 image in base64
     * @return HTTP response, can be null in case of failure
     */
    suspend fun handle(base64: String): Response?
}

open class BaseHandler(private val next: Handler?): Handler {
    /**
     * Base implementation, just calls the next handler if it exists
     * @see Handler.handle
     */
    override suspend fun handle(base64: String): Response? {
        return next?.handle(base64)
    }
}