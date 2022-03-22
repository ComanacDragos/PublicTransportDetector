package ro.ubb.mobile_app.core.detection

import android.graphics.RectF

data class DetectionResult(val boundingBox: RectF,
                           val label: String,
                           val score: Float,
                           val classIndex: Int
                           )
