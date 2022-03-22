package ro.ubb.mobile_app.live.accessible

import android.content.Context
import android.graphics.*
import android.speech.tts.TextToSpeech
import android.util.AttributeSet
import android.util.Log
import android.view.SurfaceView
import ro.ubb.mobile_app.core.TAG
import kotlin.math.sin
import kotlin.random.Random

/*
Math formulas from:
http://www.ecircuitcenter.com/Calc/draw_sine1/draw_sine_canvas_topic1.html
 */
class VoiceView(context: Context, attributeSet: AttributeSet) :
    SurfaceView(context, attributeSet) {

    private val vmax =  2
    private val tmax = 0.001
    private val noPoints = 200

    private var phase = 0
    private var peakVoltage = 0.5
    private var hz = 1000.0

    var tts: TextToSpeech? = null

    private val paint = Paint().apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = false
    }

    private val range = IntRange(0, noPoints)

    private var timer = 0
    init{
        setWillNotDraw(false)
        setZOrderOnTop(true)
    }

    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)
        if (canvas != null && tts != null) {
            canvas.drawColor(0)
            val midX = width / 2f
            val midY = height / 2f

            if (!tts!!.isSpeaking) {
                canvas.drawLine(0f, midY, width.toFloat(), midY, paint)
            } else {
                phase = (phase + 10) % width

                val xScale = (width) / (2f * tmax)
                val yScale = (height) / (2f * vmax)
                val tstart = -tmax
                val tstop = tmax

                val dt = (tstop - tstart) / (noPoints - 1)

                var lastX = -1f
                var lastY = -1f

                for (i in range) {
                    val x = tstart + i * dt

                    if (timer == 1000) {

                        timer = 0
                    }
                    timer += 1
                    peakVoltage = Random.nextDouble() / 5 + 0.5
                    hz = Random.nextDouble() * 200 + 1000

                    val y = peakVoltage * sin(2 * 3.1415 * hz * x + phase * 3.1415 / 180)

                    val xp = midX + x * xScale
                    val yp = midY - y * yScale

                    if (lastX == -1f)
                        canvas.drawPoint(xp.toFloat(), yp.toFloat(), paint)
                    else {
                        canvas.drawLine(lastX, lastY, xp.toFloat(), yp.toFloat(), paint)
                    }
                    lastX = xp.toFloat()
                    lastY = yp.toFloat()
                }
            }
            invalidate()
        }
    }
}