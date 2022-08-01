package sensors_in_paradise.sonar.custom_views.confusion_matrix

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import android.util.Log
import androidx.core.graphics.withRotation
import sensors_in_paradise.sonar.GlobalValues
import sensors_in_paradise.sonar.R
import kotlin.math.min

class DrawableConfusionMatrix(val context: Context, cm: ConfusionMatrix) : ConfusionMatrix(cm) {
    constructor(context: Context, labels: Array<String>) : this(context, ConfusionMatrix(labels))

    private class BoundedTextPaint(
        text: String,
        color: Int,
        textStyle: Int = Typeface.BOLD,
        textSize: Float = 40f
    ) : Paint(0) {
        val textBounds = Rect()

        init {
            typeface = Typeface.create(Typeface.DEFAULT, textStyle)
            this.textSize = textSize
            getTextBounds(text, 0, text.length, textBounds)
            this.color = color
        }
    }

    private val textPaintPerNumDigits = LinkedHashMap<Int, BoundedTextPaint>()

    private val labelTextPaints = LinkedHashMap<String, BoundedTextPaint>()
    private val labelAbbreviationTextPaints = LinkedHashMap<String, BoundedTextPaint>()
    private fun initLabelTextPaints() {
        labelTextPaints.clear()
        labelAbbreviationTextPaints.clear()

            for ((index, label) in labels.keys.withIndex()) {
                labelTextPaints[label] = BoundedTextPaint(label, textColor)
                labelAbbreviationTextPaints[label] = BoundedTextPaint(getAbbreviation(index), textColor)
            }
    }

    private val textColor =
        GlobalValues.getAndroidColorResource(context, android.R.attr.textColorPrimary)
    private val textColorSecondary =
        GlobalValues.getAndroidColorResource(context, android.R.attr.textColorSecondary)

    init {
        initLabelTextPaints()
    }

    fun getDesiredHeightForWidth(width: Int): Int {
        if (width == 0) {
            return 0
        }
        val cmBounds = getConfusionMatrixContentBounds(Rect(0, 0, width, width))
        val cellWidth = cmBounds.width() / getNumLabels()
           if (needsLabelAbbreviations(cellWidth)) {
               return cmBounds.height() + drawAbbreviationLegend(null, 0, 0, width)
           }
        return cmBounds.height()
    }

    private val cellPaint = Paint(0).apply {
        color = context.getColor(R.color.colorPrimary)
        alpha = 255
    }
    /**Returns bottom of where the legend ends*/
    private fun drawAbbreviationLegend(canvas: Canvas?, left: Int, top: Int, right: Int): Int {
        val entries = getLabels().mapIndexed { index, s ->
            val entry = "${getAbbreviation(index)}: $s"
            val paint = BoundedTextPaint(entry, textColor, Typeface.NORMAL, 30f)
            Pair(entry, paint)
        }

        val colWidth = entries.maxOf { it.second.textBounds.width() } + 20
        val columns = (right - left) / colWidth
        if (columns == 0) {
            return top
        }
        val rowHeight = entries.maxOf { it.second.textBounds.height() } + 20

        for ((index, entry) in entries.withIndex()) {
            val (text, paint) = entry
            val col = index % columns
            val row = index / columns

            val textWidth = paint.textBounds.width()
            val x = left + col * colWidth.toFloat() + (colWidth - textWidth) / 2
            val y = top + row * rowHeight + rowHeight.toFloat()
            canvas?.drawText(text, x, y, paint)
        }
        return top + (labels.size / columns) * rowHeight + rowHeight
    }

    private fun getAbbreviation(labelIndex: Int): String {
        var result = ""
        var index = labelIndex
        do {
            result += 'A' + index % 26
            index /= 26
        } while (index> 0)
        return result
    }

    private fun getConfusionMatrixContentBounds(cmBoundingBox: Rect): Rect {
        return Rect(cmBoundingBox).apply {
            val offset = generalAxisLabelTextBounds.height() + generalLabelTextBounds.height()
            top += offset
            left += offset
        }
    }

    private fun needsLabelAbbreviations(cellWidth: Int): Boolean {
        for (label in getLabels()) {
            val paint = labelTextPaints[label]!!
            if (paint.textBounds.width() > cellWidth) {
                return true
            }
        }
        return false
    }
    fun draw(canvas: Canvas, boundingBox: Rect) {
        val contentBounds = getConfusionMatrixContentBounds(boundingBox)
        val cellWidth = contentBounds.width() / getNumLabels()
        val cellHeight = contentBounds.width() / getNumLabels()

        val needsLabelAbbreviations = needsLabelAbbreviations(min(cellWidth, cellHeight))

        if (needsLabelAbbreviations) {
            val bottom = drawAbbreviationLegend(canvas, boundingBox.left, boundingBox.top, boundingBox.right)
            boundingBox.top = bottom + 50
        }

        drawConfusionMatrix(boundingBox, canvas, needsLabelAbbreviations)
    }

    private fun drawConfusionMatrix(boundingBox: Rect, canvas: Canvas, abbreviateLabels: Boolean) {

        val contentAndLabelsBoundingBox = Rect(boundingBox).apply {
            top += generalAxisLabelTextBounds.height()
            left += generalAxisLabelTextBounds.height()
        }

        val contentBoundingBox =
            drawContentAndLabels(contentAndLabelsBoundingBox, canvas, abbreviateLabels)
        val drawHorizontalLabel = {
            val text = ConfusionMatrix.COLUMN_AXIS_LABEL
            val paint = BoundedTextPaint(
                text,
                textColorSecondary,
                textStyle = axisTextStyle
            )
            val textBounds = paint.textBounds
            canvas.drawText(
                text,
                contentBoundingBox.centerX() - textBounds.width() / 2f,
                contentAndLabelsBoundingBox.top - generalAxisLabelTextBounds.bottom.toFloat(),
                paint
            )
        }

        val drawVerticalLabel = {
            val text = ConfusionMatrix.ROW_AXIS_LABEL
            val paint = BoundedTextPaint(
                text,
                textColorSecondary,
                textStyle = axisTextStyle
            )
            val textBounds = paint.textBounds
            val textCenterX = boundingBox.left + generalAxisLabelTextBounds.height() / 2f
            val textCenterY = contentBoundingBox.centerY().toFloat()
            canvas.withRotation(-90f, textCenterX, textCenterY) {
                canvas.drawText(
                    text,
                    textCenterX - textBounds.width() / 2f,
                    textCenterY + generalAxisLabelTextBounds.height() / 2 - generalAxisLabelTextBounds.bottom.toFloat(),
                    paint
                )
            }
        }
        drawHorizontalLabel()
        drawVerticalLabel()
    }

    /** Returns bounding box of content excluding labels*/
    private fun drawContentAndLabels(
        boundingBox: Rect,
        canvas: Canvas,
        abbreviateLabels: Boolean
    ): Rect {
        val numLabels = getNumLabels()
        val labels = getLabels()

        val textHeight = generalLabelTextBounds.height()
        val textBottomOffset = generalLabelTextBounds.bottom

        val cellWidth = ((boundingBox.width() - textHeight) / numLabels).toFloat()
        val cellHeight = ((boundingBox.height() - textHeight) / numLabels).toFloat()

        val drawColumnLabel = { col: Int, label: String, paint: Paint, textWidth: Int ->
            val startX = textHeight + boundingBox.left + cellWidth * col
            val textX = startX + (cellWidth - textWidth) / 2f
            val textY = boundingBox.top + textHeight.toFloat() - textBottomOffset
            canvas.drawText(label, textX, textY, paint)
        }
        val drawRowLabel = { row: Int, label: String, paint: Paint, textWidth: Int ->
            val startY = boundingBox.top + textHeight + cellHeight * row

            val textY = startY + (cellHeight + textHeight) / 2f - textBottomOffset
            val textX = boundingBox.left - textWidth.toFloat() / 2 + textHeight / 2

            val textCenterX = textX + textWidth / 2f
            val textCenterY = textY + textBottomOffset - textHeight / 2f

            canvas.withRotation(-90f, textCenterX, textCenterY) {
                drawText(label, textX, textY, paint)
            }
        }

        for ((i, label) in labels.withIndex()) {
            val paint = if (abbreviateLabels) labelAbbreviationTextPaints[label]!! else labelTextPaints[label]!!
            val textBounds = paint.textBounds
            val textWidth = textBounds.width()
            val text = if (abbreviateLabels) getAbbreviation(i) else label

            drawColumnLabel(i, text, paint, textWidth)
            drawRowLabel(i, text, paint, textWidth)
        }

        val contentBoundingBox = Rect(boundingBox)
        contentBoundingBox.top += textHeight
        contentBoundingBox.left += textHeight
        drawContent(contentBoundingBox, canvas)
        return contentBoundingBox
    }

    private fun drawContent(boundingBox: Rect, canvas: Canvas) {
        val maxCellValue = getMaxCellValue()
        val numLabels = getNumLabels()
        val cellWidth = (boundingBox.width() / numLabels).toFloat()
        val cellHeight = (boundingBox.height() / numLabels).toFloat()

        for (col in 0 until numLabels) {

            val x = cellWidth * col + boundingBox.left
            for (row in 0 until numLabels) {
                val y = cellHeight * row + boundingBox.top
                val cellValue = this[col, row]
                canvas.drawRect(
                    x,
                    y,
                    x + cellWidth,
                    y + cellHeight,
                    getPaintForCell(cellValue, maxCellValue)
                )
                val textPaint = getNumberTextPaint(cellValue)
                val textBounds = textPaint.textBounds
                val cellBounds =
                    Rect(x.toInt(), y.toInt(), (x + cellWidth).toInt(), (y + cellHeight).toInt())

                val textX = cellBounds.exactCenterX() - textBounds.width() / 2
                val textY = cellBounds.exactCenterY() + textBounds.height() / 2
                canvas.drawText(cellValue.toString(), textX, textY, textPaint)
            }
        }
    }

    private fun getPaintForCell(cellValue: Int, maxCellValue: Int): Paint {
        val relativeShare = if (maxCellValue != 0) ((cellValue * 255) / maxCellValue) else 0
        Log.d("DrawableConfusionMatrix", "Max cell value $maxCellValue")
        cellPaint.alpha = relativeShare
        return cellPaint
    }

    private fun getNumberTextPaint(textValue: Int): BoundedTextPaint {
        val text = textValue.toString()
        val textPaint = textPaintPerNumDigits[text.length]
        if (textPaint != null) {
            return textPaint
        }
        val newTextPaint =
            BoundedTextPaint(text, textColor, textStyle = Typeface.NORMAL)
        textPaintPerNumDigits[text.length] = newTextPaint
        return newTextPaint
    }

    companion object {
        private const val axisTextStyle = Typeface.ITALIC
        private val generalAxisLabelTextBounds =
            BoundedTextPaint("Tegt", 0, textStyle = axisTextStyle).textBounds
        private val generalLabelTextBounds = BoundedTextPaint("Tegt", 0).textBounds
    }
}
