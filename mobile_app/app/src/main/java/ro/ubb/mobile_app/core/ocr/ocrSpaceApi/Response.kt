package ro.ubb.mobile_app.core.ocr.ocrSpaceApi

/**
 * Mirrors the specification for the result from https://ocr.space/
 */
data class Response(
    val ParsedResults: List<ResponseResult>,
    val OCRExitCode: Int,
    val IsErroredOnProcessing: Boolean,
    val ProcessingTimeInMilliseconds: String
){
    /**
     * @return concatenated list of detected text results, separated by space
     */
    fun getParsedText(): String{
        return ParsedResults
            .map { it.ParsedText }
            .reduce { acc, s ->  "$acc $s"}
    }
}

/**
 * Mirrors the specification for the result from https://ocr.space/
 */
data class ResponseResult(
    val ParsedText: String,
    val ErrorMessage: String,
    val ErrorDetails: String
)