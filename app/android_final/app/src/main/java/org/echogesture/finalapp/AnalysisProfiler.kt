package org.echogesture.finalapp

/**
 * Where the analyser actually spends its time, measured rather than assumed.
 *
 * The phone recordings showed 5–10 analysed frames per second while the
 * preview looked smooth. Three causes produce that, and they have different
 * fixes:
 *
 *  1. MediaPipe detection is slow → the budget is spent in the detector.
 *  2. The YUV → Bitmap copy is slow → the budget is spent before the detector.
 *  3. The camera does not deliver faster than that → analysis is short, the gap
 *     between arrivals is long, and nothing in the app is the bottleneck.
 *
 * Total analysis time alone cannot separate these, so both stages are timed
 * separately, the arrival interval is recorded beside them, and the camera's
 * own sensor timestamps give the ground truth for how many frames the analyser
 * never got to see.
 *
 * Arrival intervals come from real wall-clock arrival, never from the synthetic
 * 33 ms MediaPipe packet timestamps, which are a constant by construction and
 * would report a 30 fps cadence however slowly frames actually arrived.
 *
 * The drop count is derived, not guessed: consecutive analysed frames carry
 * consecutive sensor timestamps, so `sensorGap / nominalInterval` is the number
 * of frames the camera produced between them. Anything above 1 was discarded by
 * KEEP_ONLY_LATEST while the analyser was still busy — the clearest single sign
 * that the analyser, not the camera, is the limit.
 *
 * Nothing here is persisted or uploaded; it is a bounded in-memory window.
 */
class AnalysisProfiler(
    private val capacity: Int = DEFAULT_CAPACITY,
    private val nominalIntervalMs: Long = 33L
) {

    private val lock = Any()
    private val convertMs = LongArray(capacity)
    private val detectMs = LongArray(capacity)
    private val intervalMs = LongArray(capacity)
    private var count = 0
    private var next = 0
    private var lastArrivalMs = 0L
    private var lastSensorMs = 0L
    private var deliveredFrames = 0L
    private var droppedFrames = 0L

    /**
     * One analysed frame, split by stage.
     *
     * [sensorTimestampNs] is the camera's own capture timestamp; a value of 0
     * means the device did not report one, in which case the drop count simply
     * stays at whatever was derived from the samples that did.
     */
    fun record(
        arrivalMs: Long,
        convertDurationMs: Long,
        detectDurationMs: Long,
        sensorTimestampNs: Long
    ) = synchronized(lock) {
        val gap = if (lastArrivalMs == 0L) 0L else arrivalMs - lastArrivalMs
        lastArrivalMs = arrivalMs

        val sensorMs = if (sensorTimestampNs > 0) sensorTimestampNs / 1_000_000 else 0L
        if (sensorMs > 0) {
            if (lastSensorMs > 0) {
                val produced = ((sensorMs - lastSensorMs) / nominalIntervalMs).coerceAtLeast(1)
                deliveredFrames += produced
                if (produced > 1) droppedFrames += produced - 1
            }
            lastSensorMs = sensorMs
        }

        convertMs[next] = convertDurationMs
        detectMs[next] = detectDurationMs
        intervalMs[next] = gap
        next = (next + 1) % capacity
        if (count < capacity) count++
    }

    fun reset() = synchronized(lock) {
        count = 0; next = 0
        lastArrivalMs = 0L; lastSensorMs = 0L
        deliveredFrames = 0L; droppedFrames = 0L
    }

    /** A stable snapshot for reporting; percentiles over the frames held. */
    fun snapshot(): Snapshot = synchronized(lock) {
        if (count == 0) {
            return Snapshot(0, 0, 0, 0, 0, 0, 0, 0, deliveredFrames, droppedFrames)
        }
        val c = LongArray(count) { convertMs[it] }
        val d = LongArray(count) { detectMs[it] }
        // The first interval is 0 by construction and would drag p50 down.
        val gaps = (1 until count).map { intervalMs[it] }.toLongArray()
        Snapshot(
            frames = count,
            convertP50 = percentile(c, 50),
            convertP95 = percentile(c, 95),
            detectP50 = percentile(d, 50),
            detectP95 = percentile(d, 95),
            intervalP50 = if (gaps.isEmpty()) 0 else percentile(gaps, 50),
            intervalP95 = if (gaps.isEmpty()) 0 else percentile(gaps, 95),
            analysisP50 = percentile(LongArray(count) { c[it] + d[it] }, 50),
            delivered = deliveredFrames,
            dropped = droppedFrames
        )
    }

    /** What the analysis path measured, with the limit attributed. */
    data class Snapshot(
        val frames: Int,
        val convertP50: Long,
        val convertP95: Long,
        val detectP50: Long,
        val detectP95: Long,
        val intervalP50: Long,
        val intervalP95: Long,
        val analysisP50: Long,
        val delivered: Long,
        val dropped: Long
    ) {
        val effectiveFps: Double
            get() = if (intervalP50 <= 0) 0.0 else 1000.0 / intervalP50

        /** Share of delivered frames the analyser managed to process. */
        val processedFraction: Double
            get() = if (delivered <= 0) 1.0 else frames.toDouble() / delivered.toDouble()

        /**
         * Which side is the limit. Deliberately conservative: the analyser is
         * blamed only when analysis occupies most of the arrival interval or
         * the camera demonstrably discarded frames. A device that simply
         * delivers few frames slowly is never reported as an app-side problem.
         */
        val limit: Limit
            get() = when {
                frames == 0 -> Limit.NO_DATA
                intervalP50 <= 0 -> Limit.NO_DATA
                analysisP50 * 100 >= intervalP50 * 80 -> Limit.ANALYSER
                dropped > 0 -> Limit.ANALYSER
                else -> Limit.CAMERA
            }

        fun summary(): String = if (frames == 0) "no frames profiled" else
            ("%.1f fps · interval p50 %d ms · analyse p50 %d ms " +
                "(convert %d / detect %d) · convert p95 %d · detect p95 %d · " +
                "delivered %d, dropped %d · limit=%s").format(
                effectiveFps, intervalP50, analysisP50,
                convertP50, detectP50, convertP95, detectP95,
                delivered, dropped, limit
            )
    }

    enum class Limit { ANALYSER, CAMERA, NO_DATA }

    companion object {
        const val DEFAULT_CAPACITY = 256

        /** Nearest-rank percentile over a copy; input is not modified. */
        fun percentile(values: LongArray, p: Int): Long {
            if (values.isEmpty()) return 0
            val sorted = values.clone()
            sorted.sort()
            val rank = (p * sorted.size + 99) / 100 - 1
            return sorted[rank.coerceIn(0, sorted.size - 1)]
        }
    }
}
