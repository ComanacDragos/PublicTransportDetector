package ro.ubb.mobile_app.core.ocr.ocrSpaceApi

data class Response(
    val ParsedResults: List<ResponseResult>,
    val OCRExitCode: Int,
    val IsErroredOnProcessing: Boolean,
    val ProcessingTimeInMilliseconds: String
)

data class ResponseResult(
    val ParsedText: String,
    val ErrorMessage: String,
    val ErrorDetails: String
)
