package ro.ubb.mobile_app.detection.configuration

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "configuration")
data class Configuration (
    var modelName: String,
    var maxNoBoxes: Int,
    var scoreThreshold: Float,
    var nmsIouThreshold: Float,
    @PrimaryKey @ColumnInfo(name = "_id") val _id: Int = 0,
)