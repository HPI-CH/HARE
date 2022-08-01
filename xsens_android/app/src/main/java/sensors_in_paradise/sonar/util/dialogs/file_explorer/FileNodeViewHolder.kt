package sensors_in_paradise.sonar.util.dialogs.file_explorer

import android.annotation.SuppressLint
import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.content.res.AppCompatResources
import com.unnamed.b.atv.model.TreeNode
import sensors_in_paradise.sonar.R

class FileNodeViewHolder(val context: Context) :
    TreeNode.BaseNodeViewHolder<FileTreeItem>(context) {
    private var toggleIV: ImageView? = null
    private val upDrawable = AppCompatResources.getDrawable(
        context,
        UP_DRAWABLE_ID
    )
    private val downDrawable = AppCompatResources.getDrawable(
        context,
        DOWN_DRAWABLE_ID
    )

    @SuppressLint("InflateParams")
    override fun createNodeView(node: TreeNode?, content: FileTreeItem?): View {
        val inflater = LayoutInflater.from(context)

        val view: View = inflater.inflate(R.layout.file_explorer_node, null, false)
        val titleTv = view.findViewById<TextView>(R.id.tv_title_fileExplorerNode)
        val sizeTv = view.findViewById<TextView>(R.id.tv_size_fileExplorerNode)
        toggleIV = view.findViewById(R.id.imageView_toggle_fileExplorerNode)
        content?.apply {
            titleTv.text = title
            sizeTv.text = size
            toggleIV?.visibility = if (content.file.isDirectory) View.VISIBLE else View.INVISIBLE
        }

        toggle(false)
        return view
    }

    override fun toggle(active: Boolean) {
        toggleIV?.setImageDrawable(
            if (active) downDrawable else upDrawable
        )
    }
    companion object {
        const val UP_DRAWABLE_ID = R.drawable.ic_baseline_keyboard_arrow_right_24
        const val DOWN_DRAWABLE_ID = R.drawable.ic_baseline_keyboard_arrow_down_24
    }
}
