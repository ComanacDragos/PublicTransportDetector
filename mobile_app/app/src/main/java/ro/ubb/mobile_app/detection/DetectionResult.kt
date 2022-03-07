package ro.ubb.mobile_app.detection

import android.graphics.RectF

data class DetectionResult(val boundingBox: RectF,
                           val label: String,
                           val score: Float,
                           val classIndex: Int
                           )
