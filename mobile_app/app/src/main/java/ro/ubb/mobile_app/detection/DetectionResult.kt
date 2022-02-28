package ro.ubb.mobile_app.detection

import android.graphics.RectF

data class DetectionResult(val boundingBox: RectF, val text: String, val classIndex: Int)
