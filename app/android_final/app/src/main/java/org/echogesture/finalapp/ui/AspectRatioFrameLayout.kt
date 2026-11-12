package org.echogesture.finalapp.ui

import android.content.Context
import android.util.AttributeSet
import android.widget.FrameLayout
import org.echogesture.finalapp.R

/**
 * FrameLayout whose height follows its width by a fixed ratio.
 *
 * The camera preview must not be `wrap_content` — PreviewView measures to zero
 * and then expands, which pushes the capture button off screen. Pinning the
 * preview to a real ratio keeps the primary action visible on every screen
 * size while still giving the signer a large viewfinder.
 */
class AspectRatioFrameLayout @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : FrameLayout(context, attrs, defStyleAttr) {

    /** height / width. 1.333f is the 3:4 portrait viewfinder. */
    private var ratio: Float = 4f / 3f

    init {
        if (attrs != null) {
            val a = context.obtainStyledAttributes(attrs, R.styleable.AspectRatioFrameLayout)
            ratio = a.getFloat(R.styleable.AspectRatioFrameLayout_heightToWidthRatio, ratio)
            a.recycle()
        }
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = MeasureSpec.getSize(widthMeasureSpec)
        val height = (width * ratio).toInt()
        super.onMeasure(
            widthMeasureSpec,
            MeasureSpec.makeMeasureSpec(height, MeasureSpec.EXACTLY)
        )
    }
}