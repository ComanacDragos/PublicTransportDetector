package ro.ubb.mobile_app

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.tabs.TabLayout
import kotlinx.android.synthetic.main.activity_main.*
import ro.ubb.mobile_app.core.SectionsPagerAdapter
import ro.ubb.mobile_app.detection.configuration.ConfigDialog

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        val sectionsPagerAdapter = SectionsPagerAdapter(this, supportFragmentManager)
        viewPager.adapter = sectionsPagerAdapter
        val tabs: TabLayout = tabs
        tabs.setupWithViewPager(viewPager)

        settingsFab.setOnClickListener {
//            val dialog = Dialog(this)
//            dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)
//            dialog.setCancelable(true)
//            dialog.setContentView(R.layout.custom_dialog)
//
//            updateButton.setOnClickListener {
//                dialog.dismiss()
//            }
//
//            dialog.show()
            ConfigDialog().show(supportFragmentManager, "ConfigDialog")
        }
    }
}