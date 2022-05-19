package ro.ubb.mobile_app.core

import android.content.Context
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentManager
import androidx.fragment.app.FragmentPagerAdapter
import ro.ubb.mobile_app.R
import ro.ubb.mobile_app.image.ImageFragment
import ro.ubb.mobile_app.live.accessible.AccessibleLiveFragment
import ro.ubb.mobile_app.live.simple.LiveFragment

private val TAB_TITLES = arrayOf(
    R.string.tab_text_image,
    R.string.tab_text_live,
    R.string.tab_text_live_accessible
)

class SectionsPagerAdapter(private val context: Context, fm: FragmentManager) : FragmentPagerAdapter(fm) {
    /**
     * @param position the page where the fragment should be displayed
     * @return which fragment to show on each page
     */
    override fun getItem(position: Int): Fragment {
        return when(position){
            0 -> ImageFragment()
            1 -> LiveFragment()
            else -> AccessibleLiveFragment()
        }
    }

    /**
     * @return the number of pages the pager has
     */
    override fun getCount(): Int {
        return 3
    }

    /**
     * @param position the page where the title is shown
     * @return title for the given page
     */
    override fun getPageTitle(position: Int): CharSequence? {
        return context.resources.getString(TAB_TITLES[position])
    }
}