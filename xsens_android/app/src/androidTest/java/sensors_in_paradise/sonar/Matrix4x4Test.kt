package sensors_in_paradise.sonar

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Before
import org.junit.Test
import sensors_in_paradise.sonar.custom_views.stickman.math.Matrix4x4
import sensors_in_paradise.sonar.custom_views.stickman.math.Vec4

class Matrix4x4Test {
    @Before
    fun init() {}

    @Test
    fun accessRowTest() {
        val m = Matrix4x4()
        val row = m.getRow(2)
        assertEquals(0f, row[0])
        assertEquals(0f, row[1])
        assertEquals(1f, row[2])
        assertEquals(0f, row[3])
    }

    @Test
    fun accessColTest() {
        val m = Matrix4x4()
        val col = m.getCol(2)
        assertEquals(0f, col[0])
        assertEquals(0f, col[1])
        assertEquals(1f, col[2])
        assertEquals(0f, col[3])
    }

    @Test
    fun accessScalarTest() {
        val m = Matrix4x4()
        assertEquals(0f, m[2, 0])
        assertEquals(0f, m[2, 1])
        assertEquals(1f, m[2, 2])
        assertEquals(0f, m[2, 3])
    }

    @Test
    fun setScalarTest() {
        val m = Matrix4x4()
        m[2, 3] = 23f
        assertEquals(23f, m[2, 3])
    }

    @Test
    fun transformVectorTest() {
        val p1 = Vec4(2f, 06f, 1f, 1f)
        val m = Matrix4x4.fromRows(
            floatArrayOf(1f, 2f, 1f, 1f),
            floatArrayOf(0f, 1f, 0f, 1f),
            floatArrayOf(2f, 3f, 4f, 1f),
            floatArrayOf(1f, 1f, 1f, 1f)
        )
        val res = m * p1
        assertEquals(16f, res.x)
        assertEquals(7f, res.y)
        assertEquals(27f, res.z)
        assertEquals(10f, res.w)
    }

    @Test
    fun transformVectorWithUnitMatrixTest() {
        val m = Matrix4x4()
        val p = Vec4(1f, 0.5f, 2f)
        val res = m * p
        assertEquals(1f, res.x)
        assertEquals(0.5f, res.y)
        assertEquals(2f, res.z)
        assertEquals(1f, res.w)
    }

    @Test
    fun instantiateUnitMatrixTest() {
        val m = Matrix4x4()
        // row 0
        assertEquals(1f, m[0, 0])
        assertEquals(0f, m[0, 1])
        assertEquals(0f, m[0, 2])
        assertEquals(0f, m[0, 3])
        // row 1
        assertEquals(0f, m[1, 0])
        assertEquals(1f, m[1, 1])
        assertEquals(0f, m[1, 2])
        assertEquals(0f, m[1, 3])
        // row 2
        assertEquals(0f, m[2, 0])
        assertEquals(0f, m[2, 1])
        assertEquals(1f, m[2, 2])
        assertEquals(0f, m[2, 3])
        // row 3
        assertEquals(0f, m[3, 0])
        assertEquals(0f, m[3, 1])
        assertEquals(0f, m[3, 2])
        assertEquals(1f, m[3, 3])
    }

    @Test
    fun matrixTimesUnitMatrixTest() {
        val m1 = Matrix4x4.fromRows(
            floatArrayOf(1f, 2f, 1f, 1f),
            floatArrayOf(0f, 1f, 0f, 1f),
            floatArrayOf(2f, 3f, 4f, 1f),
            floatArrayOf(1f, 1f, 1f, 1f)
        )
        val m2 = Matrix4x4()

        val m = m1 * m2
        // row 0
        assertEquals(1f, m[0, 0])
        assertEquals(2f, m[0, 1])
        assertEquals(1f, m[0, 2])
        assertEquals(1f, m[0, 3])
        // row 1
        assertEquals(0f, m[1, 0])
        assertEquals(1f, m[1, 1])
        assertEquals(0f, m[1, 2])
        assertEquals(1f, m[1, 3])
        // row 2
        assertEquals(2f, m[2, 0])
        assertEquals(3f, m[2, 1])
        assertEquals(4f, m[2, 2])
        assertEquals(1f, m[2, 3])
        // row 3
        assertEquals(1f, m[3, 0])
        assertEquals(1f, m[3, 1])
        assertEquals(1f, m[3, 2])
        assertEquals(1f, m[3, 3])
    }

    @Test
    fun unitMatrixTimesMatrixTest() {
        val m1 = Matrix4x4()
        val m2 = Matrix4x4.fromRows(
            floatArrayOf(1f, 2f, 1f, 1f),
            floatArrayOf(0f, 1f, 0f, 1f),
            floatArrayOf(2f, 3f, 4f, 1f),
            floatArrayOf(1f, 1f, 1f, 1f)
        )

        val res = m1 * m2
        // row 0
        assertEquals(1f, res[0, 0])
        assertEquals(2f, res[0, 1])
        assertEquals(1f, res[0, 2])
        assertEquals(1f, res[0, 3])
        // row
        assertEquals(0f, res[1, 0])
        assertEquals(1f, res[1, 1])
        assertEquals(0f, res[1, 2])
        assertEquals(1f, res[1, 3])
        // row
        assertEquals(2f, res[2, 0])
        assertEquals(3f, res[2, 1])
        assertEquals(4f, res[2, 2])
        assertEquals(1f, res[2, 3])
        // row
        assertEquals(1f, res[3, 0])
        assertEquals(1f, res[3, 1])
        assertEquals(1f, res[3, 2])
        assertEquals(1f, res[3, 3])
    }

    @Test
    fun matrixTimesMatrixTest() {
        val m1 = Matrix4x4.fromRows(
            floatArrayOf(1f, 2f, 1f, 1f),
            floatArrayOf(0f, 1f, 0f, 1f),
            floatArrayOf(2f, 3f, 4f, 1f),
            floatArrayOf(1f, 1f, 1f, 1f)
        )
        val m2 = Matrix4x4.fromRows(
            floatArrayOf(2f, 5f, 1f, 1f),
            floatArrayOf(6f, 7f, 1f, 1f),
            floatArrayOf(1f, 8f, 1f, 1f),
            floatArrayOf(1f, 1f, 1f, 1f)
        )

        val res = m1 * m2
        // row 0
        assertEquals(16f, res[0, 0])
        assertEquals(28f, res[0, 1])
        assertEquals(5f, res[0, 2])
        assertEquals(5f, res[0, 3])
        // row
        assertEquals(7f, res[1, 0])
        assertEquals(8f, res[1, 1])
        assertEquals(2f, res[1, 2])
        assertEquals(2f, res[1, 3])
        // row
        assertEquals(27f, res[2, 0])
        assertEquals(64f, res[2, 1])
        assertEquals(10f, res[2, 2])
        assertEquals(10f, res[2, 3])
        // row
        assertEquals(10f, res[3, 0])
        assertEquals(21f, res[3, 1])
        assertEquals(4f, res[3, 2])
        assertEquals(4f, res[3, 3])
    }

    @Test
    fun cloneMatrixEqualsOriginalTest() {
        val m1 = Matrix4x4.fromRows(
            floatArrayOf(1f, 2f, 1f, 1f),
            floatArrayOf(0f, 1f, 0f, 1f),
            floatArrayOf(2f, 3f, 4f, 1f),
            floatArrayOf(1f, 1f, 1f, 1f)
        )
        val res = m1.clone()
        // row 0
        assertEquals(1f, res[0, 0])
        assertEquals(2f, res[0, 1])
        assertEquals(1f, res[0, 2])
        assertEquals(1f, res[0, 3])
        // row 1
        assertEquals(0f, res[1, 0])
        assertEquals(1f, res[1, 1])
        assertEquals(0f, res[1, 2])
        assertEquals(1f, res[1, 3])
        // row 2
        assertEquals(2f, res[2, 0])
        assertEquals(3f, res[2, 1])
        assertEquals(4f, res[2, 2])
        assertEquals(1f, res[2, 3])
        // row 3
        assertEquals(1f, res[3, 0])
        assertEquals(1f, res[3, 1])
        assertEquals(1f, res[3, 2])
        assertEquals(1f, res[3, 3])
    }

    @Test
    fun matrixFromRowsTest() {
        val res = Matrix4x4.fromRows(
            floatArrayOf(1f, 2f, 1f, 1f),
            floatArrayOf(0f, 1f, 0f, 1f),
            floatArrayOf(2f, 3f, 4f, 1f),
            floatArrayOf(1f, 1f, 1f, 1f)
        )

        // row 0
        assertEquals(1f, res[0, 0])
        assertEquals(2f, res[0, 1])
        assertEquals(1f, res[0, 2])
        assertEquals(1f, res[0, 3])
        // row 1
        assertEquals(0f, res[1, 0])
        assertEquals(1f, res[1, 1])
        assertEquals(0f, res[1, 2])
        assertEquals(1f, res[1, 3])
        // row 2
        assertEquals(2f, res[2, 0])
        assertEquals(3f, res[2, 1])
        assertEquals(4f, res[2, 2])
        assertEquals(1f, res[2, 3])
        // row 3
        assertEquals(1f, res[3, 0])
        assertEquals(1f, res[3, 1])
        assertEquals(1f, res[3, 2])
        assertEquals(1f, res[3, 3])
    }

    @Test
    fun cloneMatrixIsNotSameReferenceTest() {
        val m1 = Matrix4x4.fromRows(
            floatArrayOf(1f, 2f, 1f, 1f),
            floatArrayOf(0f, 1f, 0f, 1f),
            floatArrayOf(2f, 3f, 4f, 1f),
            floatArrayOf(1f, 1f, 1f, 1f)
        )
        val res = m1.clone()

        res[0, 0] = 21f
        assertEquals(21f, res[0, 0])
        assertEquals(1f, m1[0, 0])
    }

    @Test
    fun equalityOperatorTest() {
        val m1 = Matrix4x4()
        val m2 = Matrix4x4()
        assert(m1 == m2)
        m2[2, 3] = 34f
        assertFalse(m1 == m2)
        m1[2, 3] = 34f
        assert(m1 == m2)
    }

    @Test
    fun rotateEulerXTest() {
        val m1 = Matrix4x4.rotateEuler(180f, 0f, 0f)
        val m2 = Matrix4x4().apply { rotate(180f, 1f, 0f, 0f) }
        assert(m1 == m2)
    }

    @Test
    fun rotateEulerYTest() {
        val m1 = Matrix4x4.rotateEuler(0f, 180f, 0f)
        val m2 = Matrix4x4().apply { rotate(180f, 0f, 1f, 0f) }
        assert(m1 == m2)
    }

    @Test
    fun rotateEulerZTest() {
        val m1 = Matrix4x4.rotateEuler(0f, 0f, 180f)
        val m2 = Matrix4x4().apply { rotate(180f, 0f, 0f, 1f) }
        assert(m1 == m2)
    }

    @After
    fun cleanUp() {
    }
}
