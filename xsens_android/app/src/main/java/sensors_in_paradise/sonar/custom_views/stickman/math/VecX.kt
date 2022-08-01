package sensors_in_paradise.sonar.custom_views.stickman.math

abstract class VecX<T : VecBase>(values: FloatArray, size: Int = values.size) : VecBase(values, size) {
    abstract fun clone(): T

    operator fun plus(a: Float): T {
        val res = clone()
        res += a
        return res
    }

    operator fun plus(a: VecBase): T {
        val res = clone()
        res += a
        return res
    }

    operator fun unaryMinus(): T {
        val res = clone()
        for (i in 0 until res.size) {
            res[i] = -res[i]
        }
        return res
    }

    operator fun minus(a: VecBase): T {
        val res = clone()
        res -= a
        return res
    }

    operator fun minus(a: Float): T {
        val res = clone()
        res -= a
        return res
    }

    operator fun times(a: Float): T {
        val res = clone()
        res *= a
        return res
    }

    operator fun times(a: VecBase): T {
        val res = clone()
        res *= a
        return res
    }

    operator fun div(a: VecBase): T {
        val res = clone()
        res /= a
        return res
    }

    operator fun div(a: Float): T {
        val res = clone()
        res /= a
        return res
    }

	fun assign(a: VecBase) {
        for (i in 0 until a.size.coerceAtMost(size)) {
             this[i] = a[i]
        }
    }
}
