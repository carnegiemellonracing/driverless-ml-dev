"""
Test point_to_polygonal_chain_distance() function

This tests the function that finds the minimum distance from a point
to a polygonal chain (series of connected line segments).

This function is critical for the matching algorithm (Equation 9 in paper)
where we compute M(L,R,0,0) and M(L,R,1,0) - matching points to chains.

Mathematical concept:
    dist(point, chain) = min{ dist(point, segment_i) for all segments in chain }

Test cases cover:
1. Single-segment chain (baseline)
2. Multi-segment chains (L-shape, straight path)
3. Points near vertices (where segments meet)
4. Points closest to different segments in the chain
"""

import numpy as np
import sys
sys.path.append('..')

from helper import point_to_polygonal_chain_distance


def test_single_segment_chain():
    """
    Test Case 1: Single-segment chain (simplest case)

    Setup:
        Point: (1, 1)
        Chain: [(0, 0), (2, 0)] - one horizontal segment

    Expected:
        Distance: 1.0 (perpendicular to segment)

    Why this matters:
        Validates that the function reduces to point_to_segment_distance
        when chain has only one segment. This is the baseline case.
    """
    print("\n[Test 1] Single-segment chain")

    point = np.array([1.0, 1.0])
    chain_coords = np.array([[0.0, 0.0], [2.0, 0.0]])

    distance = point_to_polygonal_chain_distance(point, chain_coords)

    expected = 1.0
    assert np.isclose(distance, expected), \
        f"Expected {expected}, got {distance}"

    print(f"  Point: {point}")
    print(f"  Chain: {chain_coords[0]} -> {chain_coords[1]}")
    print(f"  Distance: {distance:.4f}")
    print(f"  ✓ PASSED")


def test_l_shape_closest_to_first_segment():
    """
    Test Case 2: L-shaped chain - point closest to first segment

    Setup:
        Point: (1.5, 0.5)
        Chain: [(0,0), (2,0), (2,2)] - L-shaped path
               First segment: (0,0) to (2,0) [horizontal]
               Second segment: (2,0) to (2,2) [vertical]

    Expected:
        Distance: 0.5 (perpendicular to horizontal segment)

    Calculation:
        - Distance to first segment (horizontal): 0.5
        - Distance to second segment (vertical): ~1.12
        - Minimum: 0.5

    Why this matters:
        Tests that the function correctly iterates through all segments
        and finds the minimum distance (first segment in this case).
    """
    print("\n[Test 2] L-shaped chain - closest to first segment")

    point = np.array([1.5, 0.5])
    # L-shaped chain: horizontal then vertical
    chain_coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])

    distance = point_to_polygonal_chain_distance(point, chain_coords)

    expected = 0.5
    assert np.isclose(distance, expected), \
        f"Expected {expected}, got {distance}"

    print(f"  Point: {point}")
    print(f"  Chain: L-shape with 2 segments")
    print(f"    Seg 1: {chain_coords[0]} -> {chain_coords[1]} (horizontal)")
    print(f"    Seg 2: {chain_coords[1]} -> {chain_coords[2]} (vertical)")
    print(f"  Distance: {distance:.4f}")
    print(f"  ✓ PASSED")


def test_l_shape_closest_to_second_segment():
    """
    Test Case 3: L-shaped chain - point closest to second segment

    Setup:
        Point: (2.5, 1.0)
        Chain: [(0,0), (2,0), (2,2)] - same L-shape

    Expected:
        Distance: 0.5 (perpendicular to vertical segment)

    Calculation:
        - Distance to first segment: sqrt((2.5-2)^2 + (1-0)^2) = sqrt(0.25+1) ≈ 1.12
        - Distance to second segment: 0.5 (perpendicular from x=2.5 to x=2)
        - Minimum: 0.5

    Why this matters:
        Ensures the function checks ALL segments, not just the first one.
        This point is closer to the second segment.
    """
    print("\n[Test 3] L-shaped chain - closest to second segment")

    point = np.array([2.5, 1.0])
    chain_coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])

    distance = point_to_polygonal_chain_distance(point, chain_coords)

    expected = 0.5
    assert np.isclose(distance, expected), \
        f"Expected {expected}, got {distance}"

    print(f"  Point: {point}")
    print(f"  Chain: Same L-shape")
    print(f"  Distance: {distance:.4f} (to vertical segment)")
    print(f"  ✓ PASSED")


def test_point_near_vertex():
    """
    Test Case 4: Point near L-shaped chain

    Setup:
        Point: (2.1, 0.1)
        Chain: [(0,0), (2,0), (2,2)]
               Segment 1: horizontal (0,0) to (2,0)
               Segment 2: vertical (2,0) to (2,2) at x=2

    Expected:
        Distance: 0.1 (perpendicular to vertical segment)

    Calculation:
        - Distance to vertex (2,0): sqrt(0.1² + 0.1²) ≈ 0.1414
        - Distance to horizontal segment: endpoint distance ≈ 0.1414
        - Distance to vertical segment: perpendicular distance = 0.1 ✓
        - Minimum: 0.1

    Why this matters:
        Tests that function finds minimum across both vertices AND segments.
        The closest point is on the vertical segment, not the vertex.
    """
    print("\n[Test 4] Point near L-shaped chain")

    point = np.array([2.1, 0.1])
    chain_coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])

    distance = point_to_polygonal_chain_distance(point, chain_coords)

    # Closest point is on vertical segment: perpendicular distance from x=2.1 to x=2
    expected = 0.1
    assert np.isclose(distance, expected), \
        f"Expected {expected:.4f}, got {distance:.4f}"

    print(f"  Point: {point}")
    print(f"  Closest to: vertical segment at (2, 0.1)")
    print(f"  Distance: {distance:.4f}")
    print(f"  ✓ PASSED")


def test_multi_segment_straight_path():
    """
    Test Case 5: Longer straight path with multiple segments

    Setup:
        Point: (1.5, 1.0)
        Chain: [(0,0), (1,0), (2,0), (3,0)] - 3 collinear segments

    Expected:
        Distance: 1.0 (perpendicular to the chain)

    Why this matters:
        Tests that the function handles longer chains efficiently.
        All segments are collinear, so distance should be same to all.
        Point (1.5, 1.0) projects onto second segment.
    """
    print("\n[Test 5] Multi-segment straight path")

    point = np.array([1.5, 1.0])
    # Straight path with 3 segments
    chain_coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    distance = point_to_polygonal_chain_distance(point, chain_coords)

    expected = 1.0
    assert np.isclose(distance, expected), \
        f"Expected {expected}, got {distance}"

    print(f"  Point: {point}")
    print(f"  Chain: 3 collinear segments")
    print(f"  Distance: {distance:.4f}")
    print(f"  ✓ PASSED")


def test_zigzag_chain():
    """
    Test Case 6: Zigzag chain

    Setup:
        Point: (1.0, 0.5)
        Chain: [(0,0), (1,0), (1,1), (2,1)] - zigzag pattern

    Expected:
        Distance: 0.5 (to vertical segment)

    Calculation:
        - Segment 1: (0,0) to (1,0) - horizontal, distance ≈ 0.5
        - Segment 2: (1,0) to (1,1) - vertical at x=1, distance = 0.5
        - Segment 3: (1,1) to (2,1) - horizontal, distance ≈ 0.71
        - Minimum: 0.5

    Why this matters:
        Tests more complex geometry with direction changes.
        Ensures robustness with non-convex paths.
    """
    print("\n[Test 6] Zigzag chain")

    point = np.array([1.0, 0.5])
    # Zigzag: right, up, right
    chain_coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])

    distance = point_to_polygonal_chain_distance(point, chain_coords)

    # Point (1, 0.5) is on the vertical segment from (1,0) to (1,1)
    # But it's also very close to first segment at (1,0)
    # The minimum should be very small (essentially on the chain)
    # Actually, point is ON the vertical segment, so distance should be 0
    expected = 0.0
    assert np.isclose(distance, expected, atol=1e-6), \
        f"Expected {expected}, got {distance}"

    print(f"  Point: {point}")
    print(f"  Chain: Zigzag with 3 segments")
    print(f"  Distance: {distance:.6f}")
    print(f"  ✓ PASSED")


def run_all_tests():
    """Run all tests for point_to_polygonal_chain_distance"""
    print("=" * 70)
    print("Testing: point_to_polygonal_chain_distance()")
    print("=" * 70)
    print("\nThis function finds the minimum distance from a point to a")
    print("polygonal chain - used in the matching algorithm (Equation 9).")

    tests = [
        test_single_segment_chain,
        test_l_shape_closest_to_first_segment,
        test_l_shape_closest_to_second_segment,
        test_point_near_vertex,
        test_multi_segment_straight_path,
        test_zigzag_chain,
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
        print("\n✅ All tests passed! point_to_polygonal_chain_distance() works correctly.")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
