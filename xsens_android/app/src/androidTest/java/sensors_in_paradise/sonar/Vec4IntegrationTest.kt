package sensors_in_paradise.sonar

import org.junit.Assert
import org.junit.Test
import sensors_in_paradise.sonar.custom_views.stickman.math.Matrix4x4
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec4

class Vec4IntegrationTest {
    @Test
    fun timesAssignUnitMatrixTest() {
        val m = Matrix4x4()
        val v = Vec4(0.5f, 21f, 10f)

        v *= m
        Assert.assertEquals(0.5f, v[0])
        Assert.assertEquals(21f, v[1])
        Assert.assertEquals(10f, v[2])
        Assert.assertEquals(1f, v[3])
    }

    @Test
    fun timesAssignMatrixTest() {
        val m = Matrix4x4().apply { scale(2f, 1f, 1f) }
        val v = Vec4(0.5f, 21f, 10f)

        v *= m
        Assert.assertEquals(1f, v[0])
        Assert.assertEquals(21f, v[1])
        Assert.assertEquals(10f, v[2])
        Assert.assertEquals(1f, v[3])
    }

    @Test
    fun timesAssignMatrixReferenceTest() {
        val m = Matrix4x4().apply { scale(2f, 1f, 1f) }
        val v = Vec4(0.5f, 21f, 10f)
        val b = v
        v *= m

        assert(b === v)
    }
}
