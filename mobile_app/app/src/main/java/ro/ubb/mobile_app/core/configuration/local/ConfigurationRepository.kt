package ro.ubb.mobile_app.core.configuration.local

import ro.ubb.mobile_app.core.configuration.Configuration

class ConfigurationRepository(private val configurationDao: ConfigurationDao) {
    val configuration = configurationDao.getConfiguration()

    /**
     * Deletes the previous configuration from the database, then inserts the new configuration
     * @param configuration new configuration
     */
    suspend fun setConfiguration(configuration: Configuration){
        configurationDao.deleteAll()
        configurationDao.insert(configuration)
    }
}