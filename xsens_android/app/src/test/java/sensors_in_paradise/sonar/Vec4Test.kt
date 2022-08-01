package sensors_in_paradise.sonar

import org.junit.After
import org.junit.Assert
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec4

class Vec4Test {
    @Before
    fun init() {
    }

    @Test
    fun equalityTest() {
        val p1 = Vec4(1f, 0.5f, 0f)
        val p2 = Vec4(1f, 0.5f, 0f)
        val p3 = Vec4(2f, 0.5f, 0f)

        assert(p1 == p2)
        assert(p2 != p3)
        assert(p1 != p3)
    }

    @Test
    fun scalarPlusAssignTest() {
       val p1 = Vec4(1f, 0.5f, 0f)
        p1 += 2f
        assertEquals(3f, p1.x)
        assertEquals(2.5f, p1.y)
        assertEquals(2f, p1.z)
        assertEquals(3f, p1.w)
    }

    @Test
    fun vec3PlusAssignTest() {
        val p1 = Vec4(1f, 0.5f, 0f)
        val p2 = Vec4(0f, 2.5f, 1f)
        p1 += p2
        assertEquals(1f, p1.x)
        assertEquals(3f, p1.y)
        assertEquals(1f, p1.z)
        assertEquals(2f, p1.w)
    }

    @Test
    fun scalarMinusAssignTest() {
        val p1 = Vec4(1f, 0.5f, 0f)
        p1 -= 2f
        assertEquals(-1f, p1.x)
        assertEquals(-1.5f, p1.y)
        assertEquals(-2f, p1.z)
        assertEquals(-1f, p1.w)
    }

    @Test
    fun vec3MinusAssignTest() {
        val p1 = Vec4(1f, 0.5f, 0f)
        val p2 = Vec4(0f, 2.5f, 1f)
        p1 -= p2
        assertEquals(1f, p1.x)
        assertEquals(-2f, p1.y)
        assertEquals(-1f, p1.z)
        assertEquals(0f, p1.w)
    }

    @Test
    fun scalarTimesAssignTest() {
        val p1 = Vec4(1f, 0.5f, 0f)
        p1 *= 2f
        assertEquals(2f, p1.x)
        assertEquals(1f, p1.y)
        assertEquals(0f, p1.z)
        assertEquals(2f, p1.w)
    }

    @Test
    fun vec3TimesAssignTest() {
        val p1 = Vec4(1f, 0.5f, 0f)
        val p2 = Vec4(0f, 2.5f, 1f)
        p1 *= p2
        assertEquals(0f, p1.x)
        assertEquals(1.25f, p1.y)
        assertEquals(0f, p1.z)
        assertEquals(1f, p1.w)
    }

    @Test
    fun scalarDivAssignTest() {
        val p1 = Vec4(1f, 12f, 0f)
        p1 /= 2f
        assertEquals(0.5f, p1.x)
        assertEquals(6f, p1.y)
        assertEquals(0f, p1.z)
        assertEquals(0.5f, p1.w)
    }

    @Test
    fun vec3DivAssignTest() {
        val p1 = Vec4(1f, 24f, 6f)
        val p2 = Vec4(1f, 2f, 3f)
        p1 /= p2
        assertEquals(1f, p1.x)
        assertEquals(12f, p1.y)
        assertEquals(2f, p1.z)
        assertEquals(1f, p1.w)
    }

    @Test
    fun scalarPlusTest() {
        val p1 = Vec4(1f, 0.5f, 0f)
        val p2 = p1 + 2f
        assertEquals(3f, p2.x)
        assertEquals(2.5f, p2.y)
        assertEquals(2f, p2.z)
        assertEquals(3f, p2.w)
    }

    @Test
    fun vec3PlusTest() {
        val p1 = Vec4(1f, 0.5f, 0f)
        val p2 = Vec4(0f, 2.5f, 1f)
        val p3 = p1 + p2
        assertEquals(1f, p3.x)
        assertEquals(3f, p3.y)
        assertEquals(1f, p3.z)
        assertEquals(2f, p3.w)
    }

    @Test
    fun scalarMinusTest() {
        val p1 = Vec4(1f, 0.5f, 0f)
        val p2 = p1 - 2f
        assertEquals(-1f, p2.x)
        assertEquals(-1.5f, p2.y)
        assertEquals(-2f, p2.z)
        assertEquals(-1f, p2.w)
    }

    @Test
    fun vec3MinusTest() {
        val p1 = Vec4(1f, 0.5f, 0f)
        val p2 = Vec4(0f, 2.5f, 1f)
        val p3 = p1 - p2
        assertEquals(1f, p3.x)
        assertEquals(-2f, p3.y)
        assertEquals(-1f, p3.z)
        assertEquals(0f, p3.w)
    }

    @Test
    fun scalarTimesTest() {
        val p1 = Vec4(1f, 0.5f, 0f)
        val p2 = p1 * 2f
        assertEquals(2f, p2.x)
        assertEquals(1f, p2.y)
        assertEquals(0f, p2.z)
        assertEquals(2f, p2.w)
    }

    @Test
    fun vec3TimesTest() {
        val p1 = Vec4(1f, 0.5f, 0f)
        val p2 = Vec4(0f, 2.5f, 1f)
        val p3 = p1 * p2
        assertEquals(0f, p3.x)
        assertEquals(1.25f, p3.y)
        assertEquals(0f, p3.z)
        assertEquals(1f, p3.w)
    }

    @Test
    fun scalarDivTest() {
        val p1 = Vec4(1f, 12f, 0f)
        val p2 = p1 / 2f
        assertEquals(0.5f, p2.x)
        assertEquals(6f, p2.y)
        assertEquals(0f, p2.z)
        assertEquals(0.5f, p2.w)
    }

    @Test
    fun vec3DivTest() {
        val p1 = Vec4(1f, 24f, 6f)
        val p2 = Vec4(1f, 2f, 3f)
        val p3 = p1 / p2
        assertEquals(1f, p3.x)
        assertEquals(12f, p3.y)
        assertEquals(2f, p3.z)
        assertEquals(1f, p3.w)
    }

    @Test
    fun cloneTest() {
        val p1 = Vec4(1f, 0.5f, 0f, 23f)
        val p2 = p1.clone()
        assertEquals(p1.x, p2.x)
        assertEquals(p1.y, p2.y)
        assertEquals(p1.z, p2.z)
        assertEquals(p1.w, p2.w)

        p2 += 0.5f
        Assert.assertNotEquals(p1.x, p2.x)
        Assert.assertNotEquals(p1.y, p2.y)
        Assert.assertNotEquals(p1.z, p2.z)
    }

    @Test
    fun accessorTest() {
        val p1 = Vec4(1f, 0.5f, 0f, 23f)

        assertEquals(p1.x, p1[0])
        assertEquals(p1.y, p1[1])
        assertEquals(p1.z, p1[2])
        assertEquals(p1.w, p1[3])
    }

    @Test
    fun divideByZeroScalarTest() {
        val p1 = Vec4(1f, 0.5f, 2f, 23f)

        p1 /= 0f

        assertEquals(Float.POSITIVE_INFINITY, p1.x)
        assertEquals(Float.POSITIVE_INFINITY, p1.y)
        assertEquals(Float.POSITIVE_INFINITY, p1.z)
        assertEquals(Float.POSITIVE_INFINITY, p1.w)
    }

    @Test
    fun divideByZeroVec3Test() {
        val p1 = Vec4(1f, 0.5f, 2f, 23f)
        val p2 = Vec4(0f, 0f, 0f)

        p1 /= p2
        assertEquals(Float.POSITIVE_INFINITY, p1.x)
        assertEquals(Float.POSITIVE_INFINITY, p1.y)
        assertEquals(Float.POSITIVE_INFINITY, p1.z)
        assertEquals(p1[3], p1.w)
    }

    @Test
    fun xyzIsReferenceTest() {
        val p1 = Vec4(1f, 0.5f, 2f, 23f)
        val p2 = p1.xyz
        p2 += 1.0f

        assertEquals(2f, p1.x)
        assertEquals(1.5f, p1.y)
        assertEquals(3f, p1.z)
        assertEquals(23f, p1.w)
    }

    @Test
    fun vecIsReferenceInPairTest() {
        val p1 = Vec4(1f, 0.5f, 2f, 23f)
        val p2 = Vec4(1f, 0.5f, 2f, 23f)
        val arr = arrayOf(Pair(p1, p2))

        arr[0].second.x += 1f
        assertEquals(2f, p2.x)
        assertEquals(2f, arr[0].second.x)
        assertEquals(1f, p1.x)
    }

    @After
    fun cleanUp() {
    }
}
