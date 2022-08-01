package sensors_in_paradise.sonar.custom_views.stickman.math

class Vec4(values: FloatArray) : VecX<Vec4>(values, 4) {
    constructor(vec4: Vec4) : this(vec4.x, vec4.y, vec4.z, vec4.w)
    constructor(vec3: Vec3) : this(vec3.x, vec3.y, vec3.z, 1f)
    constructor(x: Float, y: Float, z: Float) : this(x, y, z, 1f)
    constructor() : this(floatArrayOf(0f, 0f, 0f, 1f))
    constructor(x: Float, y: Float, z: Float, w: Float) : this(floatArrayOf(x, y, z, w))

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

    var w: Float
        get() = this[3]
        set(value) {
            this[3] = value
        }
    var xyz: Vec3
        get() = Vec3(values)
        set(value) {
            this[0] = value[0]
            this[1] = value[1]
            this[2] = value[2]
        }
    var xy: Vec2
        get() = Vec2(values)
        set(value) {
            this[0] = value[0]
            this[1] = value[1]
        }

    override fun clone(): Vec4 {
        return Vec4(this)
    }

    operator fun timesAssign(matrix4x4: Matrix4x4) {
       val result = matrix4x4 * this
        x = result.x
        y = result.y
        z = result.z
        w = result.w
    }
}
