package sensors_in_paradise.sonar

import java.util.function.Predicate

open class ObservableArrayList<T> : ArrayList<T>() {
    private val onSizeChangedListeners = ArrayList<(size: Int)-> Unit>()
    fun addOnSizeChangedListener(listener: (size: Int) -> Unit) {
        onSizeChangedListeners.add(listener)
    }
    private val onItemAddedListener = ArrayList<(item: T, index: Int)-> Unit>()
    fun addOnItemAddedListener(listener: (item: T, index: Int) -> Unit) {
        onItemAddedListener.add(listener)
    }
    private val onItemRemovedListener = ArrayList<(item: T, index: Int)-> Unit>()
    fun addOnItemRemovedListener(listener: (item: T, index: Int) -> Unit) {
        onItemRemovedListener.add(listener)
    }

    private fun notifyObserversOfSizeChanged() {
        for (listener in onSizeChangedListeners) {
            listener(size)
        }
    }
    private fun notifyObserversOfItemAdded(item: T, index: Int) {
        for (listener in onItemAddedListener) {
            listener(item, index)
        }
    }
    private fun notifyObserversOfItemRemoved(item: T, index: Int) {
        for (listener in onItemRemovedListener) {
            listener(item, index)
        }
    }
    override fun add(element: T): Boolean {
        val result = super.add(element)
        notifyObserversOfSizeChanged()
        notifyObserversOfItemAdded(element, size - 1)
        return result
    }

    override fun add(index: Int, element: T) {
        super.add(index, element)
        notifyObserversOfSizeChanged()
        notifyObserversOfItemAdded(element, index)
    }

    override fun addAll(elements: Collection<T>): Boolean {
        val sizeBefore = size
        val result = super.addAll(elements)
        notifyObserversOfSizeChanged()
        for ((i, element) in elements.withIndex()) {
            notifyObserversOfItemAdded(element, sizeBefore + i)
        }
        return result
    }

    override fun addAll(index: Int, elements: Collection<T>): Boolean {
        val result = super.addAll(index, elements)
        notifyObserversOfSizeChanged()
        for ((i, element) in elements.withIndex()) {
            notifyObserversOfItemAdded(element, index + i)
        }
        return result
    }

    override fun remove(element: T): Boolean {
        val index = super.indexOf(element)
        val result = super.remove(element)
        notifyObserversOfSizeChanged()
        notifyObserversOfItemRemoved(element, index)
        return result
    }

    override fun removeAt(index: Int): T {
        val item = this[index]
        val result = super.removeAt(index)
        notifyObserversOfSizeChanged()
        notifyObserversOfItemRemoved(item, index)
        return result
    }

    override fun removeAll(elements: Collection<T>): Boolean {
        val removedElements = elements.filter { it in this }.map { Pair(indexOf(it), it) }
        val result = super.removeAll(elements.toSet())
        notifyObserversOfSizeChanged()
        for ((i, removedElement) in removedElements) {
            notifyObserversOfItemRemoved(removedElement, i)
        }
        return result
    }

    override fun removeIf(filter: Predicate<in T>): Boolean {
        val removedElements = this.filter { filter.test(it) }.map { Pair(indexOf(it), it) }
        val result = super.removeIf(filter)
        notifyObserversOfSizeChanged()
        for ((i, removedElement) in removedElements) {
            notifyObserversOfItemRemoved(removedElement, i)
        }
        return result
    }

    override fun removeRange(fromIndex: Int, toIndex: Int) {
        val removedElements = this.subList(fromIndex, toIndex)
        super.removeRange(fromIndex, toIndex)
        notifyObserversOfSizeChanged()
        for ((i, removedElement) in removedElements.withIndex()) {
            notifyObserversOfItemRemoved(removedElement, fromIndex + i)
        }
    }

    override fun clear() {
        val removedElements = this.toList()
        super.clear()
        notifyObserversOfSizeChanged()
        for ((i, removedElement) in removedElements.withIndex()) {
            notifyObserversOfItemRemoved(removedElement, i)
        }
    }
}
