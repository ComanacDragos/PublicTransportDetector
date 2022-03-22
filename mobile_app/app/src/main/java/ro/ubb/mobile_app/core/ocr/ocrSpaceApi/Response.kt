package ro.ubb.mobile_app.core.ocr.ocrSpaceApi

data class Response(
    val ParsedResults: List<ResponseResult>,
    val OCRExitCode: Int,
    val IsErroredOnProcessing: Boolean,
    val ProcessingTimeInMilliseconds: String
){
    fun getParsedText(): String{
        return ParsedResults
            .map { it.ParsedText }
            .reduce { acc, s ->  "$acc $s"}
    }
}

data class ResponseResult(
    val ParsedText: String,
    val ErrorMessage: String,
    val ErrorDetails: String
)