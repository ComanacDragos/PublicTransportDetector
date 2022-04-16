package ro.ubb.mobile_app.core.configuration.local

import ro.ubb.mobile_app.core.configuration.Configuration

class ConfigurationRepository(private val configurationDao: ConfigurationDao) {
    val configuration = configurationDao.getConfiguration()

    suspend fun setConfiguration(configuration: Configuration){
        configurationDao.deleteAll()
        configurationDao.insert(configuration)
    }
}