package ro.ubb.mobile_app.core.detection.configuration.local
import androidx.lifecycle.LiveData
import androidx.room.*
import ro.ubb.mobile_app.core.detection.configuration.Configuration

@Dao
interface ConfigurationDao {
    @Query("SELECT * from configuration WHERE _id=0")
    fun getConfiguration(): LiveData<Configuration>

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insert(configuration: Configuration)

    @Query("DELETE FROM configuration")
    suspend fun deleteAll()
}
