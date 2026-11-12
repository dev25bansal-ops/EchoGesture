package org.echogesture.finalapp.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import androidx.camera.view.PreviewView
import android.view.View
import androidx.core.content.ContextCompat
import org.echogesture.finalapp.R

/**
 * Framing guide drawn over the camera preview: a soft rounded rectangle with
 * four brighter corner brackets. It is guidance for a person, not a detector
 * output — it never moves and never implies the app knows where the hands are.
 */
class FrameGuideView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val stroke = context.resources.displayMetrics.density

    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeWidth = 3f * stroke
        color = ContextCompat.getColor(context, R.color.text_on_dark)
    }

    private val framePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * stroke
        color = 0x33FFFFFF
    }

    private val rect = RectF()
    private val path = Path()
    private var previewView: PreviewView? = null

    /** Lets the guide size its brackets to the actual preview image area. */
    fun attachTo(preview: PreviewView) {
        previewView = preview
        preview.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ -> invalidate() }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width == 0 || height == 0) return

        // The guide sits on the preview view itself. PreviewView crops with
        // FILL_CENTER, so the visible pixels cover the whole view and a
        // percentage inset is an honest framing target.
        val insetX = width * 0.10f
        val insetY = height * 0.12f
        rect.set(insetX, insetY, width - insetX, height - insetY)
        if (rect.width() <= 0f || rect.height() <= 0f) return
        val radius = 24f * stroke

        canvas.drawRoundRect(rect, radius, radius, framePaint)

        val arm = minOf(rect.width(), rect.height()) * 0.16f
        val r = radius

        // Top-left
        path.reset()
        path.moveTo(rect.left, rect.top + arm)
        path.lineTo(rect.left, rect.top + r)
        path.quadTo(rect.left, rect.top, rect.left + r, rect.top)
        path.lineTo(rect.left + arm, rect.top)
        canvas.drawPath(path, cornerPaint)

        // Top-right
        path.reset()
        path.moveTo(rect.right - arm, rect.top)
        path.lineTo(rect.right - r, rect.top)
        path.quadTo(rect.right, rect.top, rect.right, rect.top + r)
        path.lineTo(rect.right, rect.top + arm)
        canvas.drawPath(path, cornerPaint)

        // Bottom-right
        path.reset()
        path.moveTo(rect.right, rect.bottom - arm)
        path.lineTo(rect.right, rect.bottom - r)
        path.quadTo(rect.right, rect.bottom, rect.right - r, rect.bottom)
        path.lineTo(rect.right - arm, rect.bottom)
        canvas.drawPath(path, cornerPaint)

        // Bottom-left
        path.reset()
        path.moveTo(rect.left + arm, rect.bottom)
        path.lineTo(rect.left + r, rect.bottom)
        path.quadTo(rect.left, rect.bottom, rect.left, rect.bottom - r)
        path.lineTo(rect.left, rect.bottom - arm)
        canvas.drawPath(path, cornerPaint)
    }
}