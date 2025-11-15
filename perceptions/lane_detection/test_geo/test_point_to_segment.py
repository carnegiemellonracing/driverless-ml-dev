"""
Test point_to_segment_distance() function

This tests the fundamental geometric function that calculates
the minimum distance from a point to a line segment.

Mathematical cases tested:
1. Projection onto segment (0 < t < 1) - perpendicular distance
2. Projection before start (t ≤ 0) - distance to start point
3. Projection after end (t ≥ 1) - distance to end point
4. Degenerate segment (start == end) - point-to-point distance
5. Diagonal segment - non-axis-aligned geometry
"""

import numpy as np
import sys
sys.path.append('..')

from helper import point_to_segment_distance


def test_perpendicular_projection():
    """
    Test Case 1: Point projects onto segment (main case)

    Setup:
        Point: (1, 1)
        Segment: (0, 0) to (2, 0) [horizontal line]

    Expected:
        - Projection point: (1, 0)
        - Distance: 1.0 (perpendicular)

    Why this matters:
        This is the most common case - tests that the function
        correctly computes perpendicular distance when the
        projection falls within the segment.
    """
    print("\n[Test 1] Perpendicular projection onto segment")

    point = np.array([1.0, 1.0])
    seg_start = np.array([0.0, 0.0])
    seg_end = np.array([2.0, 0.0])

    distance = point_to_segment_distance(point, seg_start, seg_end)

    expected = 1.0
    assert np.isclose(distance, expected), \
        f"Expected {expected}, got {distance}"

    print(f"  Point: {point}")
    print(f"  Segment: {seg_start} to {seg_end}")
    print(f"  Distance: {distance:.4f}")
    print(f"  ✓ PASSED")


def test_projection_before_start():
    """
    Test Case 2: Point projects before segment start

    Setup:
        Point: (-1, 0)
        Segment: (0, 0) to (2, 0)

    Expected:
        - Closest point on segment: (0, 0) [start point]
        - Distance: 1.0

    Why this matters:
        Tests edge case where the perpendicular projection
        would fall outside the segment (before start).
        Function should use distance to start point instead.
    """
    print("\n[Test 2] Projection before segment start")

    point = np.array([-1.0, 0.0])
    seg_start = np.array([0.0, 0.0])
    seg_end = np.array([2.0, 0.0])

    distance = point_to_segment_distance(point, seg_start, seg_end)

    expected = 1.0
    assert np.isclose(distance, expected), \
        f"Expected {expected}, got {distance}"

    print(f"  Point: {point}")
    print(f"  Segment: {seg_start} to {seg_end}")
    print(f"  Distance: {distance:.4f}")
    print(f"  ✓ PASSED")


def test_projection_after_end():
    """
    Test Case 3: Point projects after segment end

    Setup:
        Point: (3, 0)
        Segment: (0, 0) to (2, 0)

    Expected:
        - Closest point on segment: (2, 0) [end point]
        - Distance: 1.0

    Why this matters:
        Tests edge case where the perpendicular projection
        would fall outside the segment (after end).
        Function should use distance to end point instead.
    """
    print("\n[Test 3] Projection after segment end")

    point = np.array([3.0, 0.0])
    seg_start = np.array([0.0, 0.0])
    seg_end = np.array([2.0, 0.0])

    distance = point_to_segment_distance(point, seg_start, seg_end)

    expected = 1.0
    assert np.isclose(distance, expected), \
        f"Expected {expected}, got {distance}"

    print(f"  Point: {point}")
    print(f"  Segment: {seg_start} to {seg_end}")
    print(f"  Distance: {distance:.4f}")
    print(f"  ✓ PASSED")


def test_degenerate_segment():
    """
    Test Case 4: Degenerate segment (start == end)

    Setup:
        Point: (1, 1)
        Segment: (0, 0) to (0, 0) [zero-length segment]

    Expected:
        - Distance: √2 ≈ 1.414

    Why this matters:
        Tests numerical stability - ensures no division by zero.
        When segment has zero length, should compute point-to-point
        distance.
    """
    print("\n[Test 4] Degenerate segment (zero length)")

    point = np.array([1.0, 1.0])
    seg_start = np.array([0.0, 0.0])
    seg_end = np.array([0.0, 0.0])

    distance = point_to_segment_distance(point, seg_start, seg_end)

    expected = np.sqrt(2)  # √(1² + 1²)
    assert np.isclose(distance, expected), \
        f"Expected {expected}, got {distance}"

    print(f"  Point: {point}")
    print(f"  Segment: {seg_start} to {seg_end} (degenerate)")
    print(f"  Distance: {distance:.4f}")
    print(f"  Expected: {expected:.4f}")
    print(f"  ✓ PASSED")


def test_diagonal_segment():
    """
    Test Case 5: Diagonal segment (non-axis-aligned)

    Setup:
        Point: (0, 1)
        Segment: (0, 0) to (1, 1) [diagonal line y=x]

    Expected:
        - Projection point: (0.5, 0.5) [midpoint]
        - Distance: √0.5 ≈ 0.707

    Calculation:
        - Line is y = x
        - Perpendicular from (0, 1) to y=x hits at (0.5, 0.5)
        - Distance = √((0-0.5)² + (1-0.5)²) = √(0.25 + 0.25) = √0.5

    Why this matters:
        Tests that the function works correctly for non-axis-aligned
        segments, which is critical for real-world cone positions.
    """
    print("\n[Test 5] Diagonal segment (non-axis-aligned)")

    point = np.array([0.0, 1.0])
    seg_start = np.array([0.0, 0.0])
    seg_end = np.array([1.0, 1.0])

    distance = point_to_segment_distance(point, seg_start, seg_end)

    expected = np.sqrt(0.5)  # √0.5 ≈ 0.707
    assert np.isclose(distance, expected), \
        f"Expected {expected}, got {distance}"

    print(f"  Point: {point}")
    print(f"  Segment: {seg_start} to {seg_end} (diagonal)")
    print(f"  Distance: {distance:.4f}")
    print(f"  Expected: {expected:.4f}")
    print(f"  ✓ PASSED")


def run_all_tests():
    """Run all tests for point_to_segment_distance"""
    print("=" * 70)
    print("Testing: point_to_segment_distance()")
    print("=" * 70)
    print("\nThis function calculates the minimum distance from a point")
    print("to a line segment - a fundamental operation for lane detection.")

    tests = [
        test_perpendicular_projection,
        test_projection_before_start,
        test_projection_after_end,
        test_degenerate_segment,
        test_diagonal_segment,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✅ All tests passed! point_to_segment_distance() works correctly.")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
