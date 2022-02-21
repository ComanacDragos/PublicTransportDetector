package ro.ubb.mobile_app.core

import android.content.Context
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentManager
import androidx.fragment.app.FragmentPagerAdapter
import ro.ubb.mobile_app.R
import ro.ubb.mobile_app.image.ImageFragment
import ro.ubb.mobile_app.video.VideoFragment

private val TAB_TITLES = arrayOf(
    R.string.tab_text_image,
    R.string.tab_text_video
)
/**
 * A [FragmentPagerAdapter] that returns a fragment corresponding to
 * one of the sections/tabs/pages.
 */
class SectionsPagerAdapter(private val context: Context, fm: FragmentManager) : FragmentPagerAdapter(fm) {

    override fun getItem(position: Int): Fragment {
        // getItem is called to instantiate the fragment for the given page.
        // Return a PlaceholderFragment (defined as a static inner class below).
        if(position == 0)
            return ImageFragment()
        return VideoFragment()
    }

    override fun getCount(): Int {
        // Show 2 total pages.
        return 2
    }

    override fun getPageTitle(position: Int): CharSequence? {
        return context.resources.getString(TAB_TITLES[position])
    }
}