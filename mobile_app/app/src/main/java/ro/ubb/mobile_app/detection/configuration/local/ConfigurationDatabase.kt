package ro.ubb.mobile_app.detection.configuration.local

import android.content.Context
import android.util.Log
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.sqlite.db.SupportSQLiteDatabase
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import ro.ubb.mobile_app.core.TAG
import ro.ubb.mobile_app.detection.configuration.Configuration
import androidx.room.migration.Migration




@Database(entities = [Configuration::class], version = 2)
abstract class ConfigurationDatabase: RoomDatabase() {
    abstract fun configurationDao(): ConfigurationDao

    companion object {
        @Volatile
        private var INSTANCE: ConfigurationDatabase? = null

        //        @kotlinx.coroutines.InternalCoroutinesApi()
        fun getDatabase(context: Context, scope: CoroutineScope): ConfigurationDatabase {
            val inst = INSTANCE
            if (inst != null) {
                return inst
            }
            val instance =
                Room.databaseBuilder(
                    context.applicationContext,
                    ConfigurationDatabase::class.java,
                    "conf_db"
                )
                    .addCallback(WordDatabaseCallback(scope))
                    .fallbackToDestructiveMigration()
                    .build()
            INSTANCE = instance
            return instance
        }

        private class WordDatabaseCallback(private val scope: CoroutineScope) :
            RoomDatabase.Callback() {

            override fun onOpen(db: SupportSQLiteDatabase) {
                super.onOpen(db)
                INSTANCE?.let { database ->
                    scope.launch(Dispatchers.IO) {
                        //database.configurationDao().deleteAll()
                        try{
                            database.configurationDao().insert(
                                Configuration(
                                    "model_v26.tflite",
                                    5,
                                    50f,
                                    50f
                                )
                            )
                            Log.v(TAG, "initialize database")
                        }catch (error: Exception){
                            Log.v(TAG, "ERROR: $error")
                        }


                    }
                }
            }
        }
    }
}