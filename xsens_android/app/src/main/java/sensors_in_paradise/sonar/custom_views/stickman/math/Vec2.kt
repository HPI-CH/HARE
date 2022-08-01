package sensors_in_paradise.sonar.custom_views.stickman.math

class Vec2(values: FloatArray) : VecX<Vec2>(values, 2) {
    constructor(vec2: Vec2) : this(vec2.x, vec2.y)
    constructor(x: Float, y: Float) : this(floatArrayOf(x, y))

    var x: Float
        get() = this[0]
        set(value) {
            this[0] = value
        }
    var y: Float
        get() = this[1]
        set(value) {
            this[1] = value
        }

    override fun clone(): Vec2 {
        return Vec2(this)
    }
}
