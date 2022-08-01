package sensors_in_paradise.sonar.util.dialogs.file_explorer

import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.TextView
import com.unnamed.b.atv.model.TreeNode
import com.unnamed.b.atv.view.AndroidTreeView
import sensors_in_paradise.sonar.R
import java.io.File

@SuppressLint("InflateParams")
class FileExplorerDialog(context: Context, rootDir: File, title: String = rootDir.name) {

    init {
        val builder: AlertDialog.Builder = AlertDialog.Builder(context)
        builder.setTitle(title)
        val containerView =
            LayoutInflater.from(context).inflate(R.layout.file_explorer_dialog, null) as FrameLayout
        val noFilesTv =
            containerView.findViewById<TextView>(R.id.textView_noFiles_fileExplorerDialog)

        val root: TreeNode = TreeNode.root()

        val itemsAdded = addChildrenToDir(rootDir, root, maxDepth = 5)
        if (itemsAdded > 0) {
            noFilesTv.visibility = View.INVISIBLE
        }
        val tView = AndroidTreeView(context, root)
        tView.setDefaultViewHolder(FileNodeViewHolder::class.java)
        tView.setDefaultContainerStyle(R.style.TreeNodeStyleCustom)
        containerView.addView(tView.view.apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        })
        builder.setView(containerView)
        val dialog = builder.create()
        dialog.show()
    }

    private fun addChildrenToDir(
        dirFile: File,
        dirNode: TreeNode,
        maxDepth: Int = Integer.MAX_VALUE
    ): Int {
        if (maxDepth <= 0) {
            return 0
        }
        var result = 0
        val children = dirFile.listFiles()
        if (children != null) {
            for (child in children) {
                val childNode = TreeNode(FileTreeItem(child))
                dirNode.addChild(childNode)
                result += 1
                if (child.isDirectory) {
                    result += addChildrenToDir(child, childNode, maxDepth - 1)
                }
            }
        }
        return result
    }
}
