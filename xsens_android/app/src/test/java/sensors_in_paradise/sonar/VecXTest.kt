package sensors_in_paradise.sonar

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec3
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec4

class VecXTest {
    @Before
    fun init() {
    }

    @Test
    fun assignSameDimVec() {
        val a = Vec4(0.5f, 1f, 2f)
        val b = Vec4(2f, 3f, 4f)

        a.assign(b)
        assertEquals(2f, a.x)
        assertEquals(3f, a.y)
        assertEquals(4f, a.z)
    }

    @Test
    fun assignHigherDimVec() {
        val a = Vec3(0.5f, 1f, 2f)
        val b = Vec4(2f, 3f, 4f)

        a.assign(b)
        assertEquals(2f, a.x)
        assertEquals(3f, a.y)
        assertEquals(4f, a.z)
    }

    @Test
    fun assignLowerDimVec() {
        val a = Vec3(0.5f, 1f, 2f)
        val b = Vec4(2f, 3f, 4f, 4f)

        b.assign(a)
        assertEquals(0.5f, b.x)
        assertEquals(1f, b.y)
        assertEquals(2f, b.z)
        assertEquals(4f, b.w)
    }

    @After
    fun cleanUp() {
    }
}
