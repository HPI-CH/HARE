package sensors_in_paradise.sonar.custom_views.stickman.math

open class Vec3(values: FloatArray) : VecX<Vec3>(values, 3) {
    constructor(vec3: Vec3) : this(vec3.x, vec3.y, vec3.z)
    constructor() : this(0f, 0f, 0f)
    constructor(x: Float, y: Float, z: Float) : this(floatArrayOf(x, y, z))

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
    var z: Float
        get() = this[2]
        set(value) {
            this[2] = value
        }

    override fun clone(): Vec3 {
        return Vec3(this)
    }
}
