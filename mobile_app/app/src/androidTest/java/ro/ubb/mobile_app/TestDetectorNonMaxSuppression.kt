package ro.ubb.mobile_app

import android.graphics.RectF
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.mockk.every
import io.mockk.mockkObject
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import ro.ubb.mobile_app.core.configuration.Configuration
import ro.ubb.mobile_app.core.detection.DetectionResult
import ro.ubb.mobile_app.core.detection.Detector
import java.util.*

@RunWith(AndroidJUnit4::class)
class TestDetectorNonMaxSuppression {
    @Before
    fun setup(){
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        mockkObject(MainActivity)
        every { MainActivity.applicationContext() } returns appContext

        if(!Detector.isDetectorInitialized())
            Detector.setConfiguration(
                Configuration(
                    "ssd_mobilenet.tflite",
                    20,
                    0.5f,
                    0.5f
                )
            )
    }

    @Test
    fun fullyOverlappingBoxesIou() {
        val iou = Detector.intersectionOverUnion(RectF(
            10f,10f,20f, 20f
        ), RectF(10f, 10f, 20f, 20f))

        assertEquals(1f, iou)
    }

    @Test
    fun partiallyXAxisOverlappingBoxesIou() {
        val iou = Detector.intersectionOverUnion(RectF(
            10f,10f,20f, 20f
        ), RectF(15f, 10f, 25f, 20f))

        assertEquals(1/3f, iou)
    }

    @Test
    fun partiallyYAxisOverlappingBoxesIou() {
        val iou = Detector.intersectionOverUnion(RectF(
            10f,10f,20f, 20f
        ), RectF(10f, 15f, 20f, 25f))

        assertEquals(1/3f, iou)
    }

    @Test
    fun notOverlappingBoxesIou() {
        val iou = Detector.intersectionOverUnion(
            RectF(
            10f,10f,20f, 20f
        ), RectF(20f, 20f, 30f, 30f)
        )

        assertEquals(0f, iou)
    }


    @Test
    fun emptyList(){
        val actual = Detector.nonMaximumSuppression(LinkedList<DetectionResult>()).size

        assertEquals(0, actual)
    }

    @Test
    fun oneBox(){
        val input =  listOf(
            DetectionResult(
                RectF(1f,1f,2f,2f),
                "label", 0.5f, 0)
        )

        val actual = Detector.nonMaximumSuppression(input)

        assertEquals(1, actual.size)
        assertEquals(input[0], actual[0])
    }

    @Test
    fun twoBoxesDifferentLabels(){
        val input =  listOf(
            DetectionResult(
                RectF(1f,1f,2f,2f),
                "label", 0.5f, 0),
            DetectionResult(
                RectF(1f,1f,2f,2f),
                "label2", 0.5f, 1)
        )

        val actual = Detector.nonMaximumSuppression(input)

        assertEquals(2, actual.size)
        for(i in input.indices)
            assertEquals(input[i], actual[i])
    }

    @Test
    fun twoBoxesSameLabelsNoOverlap(){
        val input =  listOf(
            DetectionResult(
                RectF(1f,1f,2f,2f),
                "label", 0.5f, 0),
            DetectionResult(
                RectF(2f,2f,3f,3f),
                "label", 0.5f, 0)
        )

        val actual = Detector.nonMaximumSuppression(input)

        assertEquals(2, actual.size)
        for(i in input.indices)
            assertEquals(input[i], actual[i])
    }

    @Test
    fun twoBoxesSameLabelsWithOverlap(){
        val input =  listOf(
            DetectionResult(
                RectF(1f,1f,2f,2f),
                "label", 0.5f, 0),
            DetectionResult(
                RectF(1f,1f,2f,2f),
                "label", 0.7f, 0)
        )

        val actual = Detector.nonMaximumSuppression(input)

        assertEquals(1, actual.size)
        assertEquals(input[1], actual[0])
    }
}