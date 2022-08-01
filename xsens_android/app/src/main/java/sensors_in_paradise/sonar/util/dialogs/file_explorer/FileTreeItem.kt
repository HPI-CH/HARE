package sensors_in_paradise.sonar.util.dialogs.file_explorer

import sensors_in_paradise.sonar.GlobalValues
import java.io.File
import java.text.CharacterIterator
import java.text.StringCharacterIterator

class FileTreeItem(val file: File) {
    private val icon = GlobalValues.getFileEmoji(file)
    val title = "$icon ${file.name}"
    val size = if (file.isFile) humanReadableByteCountSI(file.length()) else ""

    companion object {
        fun humanReadableByteCountSI(byteSize: Long): String? {
            var bytes = byteSize
            if (-1000 < bytes && bytes < 1000) {
                return "$bytes B"
            }
            val ci: CharacterIterator = StringCharacterIterator("kMGTPE")
            while (bytes <= -999950 || bytes >= 999950) {
                bytes /= 1000
                ci.next()
            }
            return java.lang.String.format("%.1f %cB", bytes / 1000.0, ci.current())
        }
    }
}
