#!/usr/bin/env python3
import unittest
import numpy as np
from python.grade_results import get_integral_from_line, compute_AC_curve, get_streak


class TestGetIntegralFromLine(unittest.TestCase):
    # Unit test to get an integral from given values
    # step corrensponds to constant multiplication of the x in the integral eg.: y = step * x
    def test_normal_cases(self):
        # Test 1: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 with step 1
        # Test 2: 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 with step 1
        # Test 3: 1, 2, 3, 4, 5 with step 1
        # Test 4: 0, 1, 2, 3, 4 with step 1
        i1 = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        i2 = [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        i3 = [[1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]]
        i4 = [[0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5]]
        t1 = get_integral_from_line(i1)
        t2 = get_integral_from_line(i2)
        t3 = get_integral_from_line(i3)
        t4 = get_integral_from_line(i4)
        self.assertEqual(t1, 49.5, "incorrect forward")
        self.assertEqual(t2, 49.5, "incorrect backward")
        self.assertEqual(t3, 12, "incorrect short")
        self.assertEqual(t4, 8, "incorrect start")

    def test_zero_case(self):
        # Test 1: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1 with step 1
        i1 = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
        t1 = get_integral_from_line(i1)
        self.assertEqual(t1, 0, "incorrect zero output")

    def test_empty_case(self):
        # Test 1: [[].[]]
        t1 = get_integral_from_line([[], []])
        self.assertEqual(t1, 0, "incorrect empty output")

    def test_negative_case(self):
        # Test 1: -1, -2, -3, -4, -5, -6, -7, -8, -9, -10 with step 1
        i1 = [[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        t1 = get_integral_from_line(i1)
        self.assertEqual(t1, -49.5, "incorrect negative output")

    def test_float_case(self):
        # Test 1: 1.1, 2.2, 3.3, 4.4, 5.5 with step 1
        i1 = [[1.1, 2.2, 3.3, 4.4, 5.5],
              [1, 2, 3, 4, 5]]
        t1 = get_integral_from_line(i1)
        self.assertEqual(t1, 13.2, "incorrect float output")

    def test_step_two_case(self):
        # Test 1: 1, 2, 3, 4, 5 with step 2
        # Test 2: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 with step 2
        # Test 3: -1, -2, -3, -4, -5, -6, -7, -8, -9, -10 with step 2
        i1 = [[1, 2, 3, 4, 5],
              [1, 3, 5, 7, 9]]
        i2 = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]]
        i3 = [[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
                [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]]
        t1 = get_integral_from_line(i1)
        t2 = get_integral_from_line(i2)
        t3 = get_integral_from_line(i3)
        self.assertEqual(t1, 24, "incorrect step two output")
        self.assertEqual(t2, 99, "incorrect step two output")
        self.assertEqual(t3, -99, "incorrect step two output")

    def test_step_three_case(self):
        # Test 1: 1, 2, 3, 4 with step 3
        # Test 2: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 with step 3
        # Test 3: -1, -2, -3, -4, -5, -6, -7, -8, -9, -10 with step 3
        i1 = [[1, 2, 3, 4],
              [1, 4, 7, 10]]
        i2 = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]]
        i3 = [[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
                [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]]
        t1 = get_integral_from_line(i1)
        t2 = get_integral_from_line(i2)
        t3 = get_integral_from_line(i3)
        self.assertEqual(t1, 22.5, "incorrect step three output")
        self.assertEqual(t2, 148.5, "incorrect step three output")
        self.assertEqual(t3, -148.5, "incorrect step three output")

    def test_step_irregular_case(self):
        # Test 1: 1, 2, 3, 4, 5 with step 1, 2, 4, 6, 8
        i1 = [[1, 2, 3, 4, 5],
                [1, 2, 4, 6, 8]]
        t1 = get_integral_from_line(i1)
        self.assertEqual(t1, 22.5, "incorrect irregular step output")

    def test_float_step_case(self):
        # Test 1: 1, 2, 3, 4, 5 with step 0.5
        i1 = [[1, 2, 3, 4, 5],
                [1, 1.5, 2, 2.5, 3]]
        t1 = get_integral_from_line(i1)
        self.assertEqual(t1, 6, "incorrect float step output")


class TestComputeACCurve(unittest.TestCase):
    # Unit test to for a computation of AC curve
    def test_normal_case(self):
        # Test 1: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        # Test 2: 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
        i1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        r1 = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        t1 = compute_AC_curve(np.array(i1))
        i2 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        r2 = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        t2 = compute_AC_curve(np.array(i2))

        np.testing.assert_allclose(t1, t2, err_msg="ac curve not agnostic of diection")
        np.testing.assert_allclose(t1, r1, err_msg="ac curve not agnostic of diection")
        np.testing.assert_allclose(t1, r2, err_msg="ac curve not agnostic of diection")

    def test_negative_case(self):
        # Test 1: -1, -2, -3, -4, -5, -6, -7, -8, -9, -10
        i1 = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
        r1 = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        t1 = compute_AC_curve(np.array(i1))
        np.testing.assert_allclose(t1, r1, err_msg="ac curve not working for negrative values")

    def test_zero_case(self):
        # Test 1: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        i1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        r1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        t1 = compute_AC_curve(np.array(i1))
        np.testing.assert_allclose(t1, r1, err_msg="ac curve not working for zero values")

    def test_empty_case(self):
        # Test 1: []
        i1 = []
        r1 = [[0], [0]]
        t1 = compute_AC_curve(np.array(i1))
        np.testing.assert_allclose(t1, r1, err_msg="ac curve not working for empty values")

    def test_float_case(self):
        # Test 1: 1.1, 2.2, 3.3, 4.4, 5.5
        i1 = [1.1, 2.2, 3.3, 4.4, 5.5]
        r1 = [[1.1, 2.2, 3.3, 4.4, 5.5], [0.0, 0.2, 0.4, 0.6, 0.8]]
        t1 = compute_AC_curve(np.array(i1))
        np.testing.assert_allclose(t1, r1, err_msg="ac curve not working for float values")

    def test_negative_float_case(self):
        # Test 1: -1.1, -2.2, -3.3, -4.4, -5.5
        i1 = [-1.1, -2.2, -3.3, -4.4, -5.5]
        r1 = [[1.1, 2.2, 3.3, 4.4, 5.5], [0.0, 0.2, 0.4, 0.6, 0.8]]
        t1 = compute_AC_curve(np.array(i1))
        np.testing.assert_allclose(t1, r1, err_msg="ac curve not working for negative float values")

    def test_mixed_case(self):
        # Test 1: 1.1, -2.2, 3.3, -4.4, 5.5
        i1 = [1.1, -2.2, 3.3, -4.4, 5.5]
        r1 = [[1.1, 2.2, 3.3, 4.4, 5.5], [0.0, 0.2, 0.4, 0.6, 0.8]]
        t1 = compute_AC_curve(np.array(i1))
        np.testing.assert_allclose(t1, r1, err_msg="ac curve not working for mixed values")

    def test_out_of_order_case(self):
        # Test 1: 1.1, 3.3, -2.2, 5.5, -4.4
        i1 = [1.1, 3.3, -2.2, 5.5, -4.4]
        r1 = [[1.1, 2.2, 3.3, 4.4, 5.5], [0.0, 0.2, 0.4, 0.6, 0.8]]
        t1 = compute_AC_curve(np.array(i1))
        np.testing.assert_allclose(t1, r1, err_msg="ac curve not working for out of order values")

    def test_repeating_case(self):
        # Test 1: 1.1, 4.4, 2.2, 4.4, 5.5
        i1 = [1.1, 4.4, 2.2, 4.4, 5.5]
        r1 = [[1.1, 2.2, 4.4, 4.4, 5.5], [0.0, 0.2, 0.4, 0.6, 0.8]]
        t1 = compute_AC_curve(np.array(i1))
        np.testing.assert_allclose(t1, r1, err_msg="ac curve not working for repeating values")


if __name__ == '__main__':
    unittest.main()
