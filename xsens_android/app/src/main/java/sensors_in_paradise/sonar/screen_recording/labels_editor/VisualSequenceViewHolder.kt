package sensors_in_paradise.sonar.screen_recording.labels_editor

abstract class VisualSequenceViewHolder(
    private val onSourceLoadedListener: () -> Unit,
    protected val onStartLoadingSource: () -> Unit,
    private val onSeekToNewPosition: ((ms: Long) -> Unit)? = null
) : IntervalLoopSeeker() {
    var sourcePath: String? = null
    var isSourceLoaded = false
    fun loadSource() {
        if (sourcePath != null) {
            onStartLoadingSource()
            loadSource(sourcePath!!) {
                isSourceLoaded = true
                onSourceLoadedListener()
            }
        }
    }

    protected abstract fun loadSource(sourcePath: String, onSourceLoadedListener: () -> Unit)

    override fun seekTo(ms: Long) {
        onSeekToNewPosition?.let { it(ms) }
    }

    companion object {
        const val FPS = 24L
    }
}
