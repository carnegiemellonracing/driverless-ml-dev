import os
import yaml
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geo import constraint_decider, bt_decider, find_matching_points
from angle_utils import calculate_segment_angle

class RealTrackDeciderTester:
    """
    Test constraint_decider and bt_decider with real track data only.
    """
    
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            self.dataset_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/dataset"
        else:
            self.dataset_path = dataset_path
        
        self.tracks = []
        self.load_all_tracks()
    
    def load_yaml_data(self, path):
        """Load YAML data from file"""
        with open(path, 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    
    def load_all_tracks(self):
        """Load all boundary and cone map data"""
        boundary_paths = [f"{self.dataset_path}/boundaries_{i}.yaml" for i in range(1, 10)]
        cone_map_paths = [f"{self.dataset_path}/cone_map_{i}.yaml" for i in range(1, 10)]
        
        self.tracks = []
        for i, (boundary_path, cone_map_path) in enumerate(zip(boundary_paths, cone_map_paths)):
            if os.path.exists(boundary_path) and os.path.exists(cone_map_path):
                boundaries = self.load_yaml_data(boundary_path)
                cone_map = self.load_yaml_data(cone_map_path)
                
                # Convert cone map to coordinate lookup
                coordinates = {}
                for point_id, coords in cone_map.items():
                    coordinates[int(point_id)] = coords
                
                track_data = {
                    'id': i + 1,
                    'boundaries': boundaries,
                    'coordinates': coordinates,
                    'left_points': [coordinates[point_id] for point_id in boundaries['left'] if point_id in coordinates],
                    'right_points': [coordinates[point_id] for point_id in boundaries['right'] if point_id in coordinates]
                }
                
                self.tracks.append(track_data)
    
    def test_track(self, track_id, num_points=10):
        """Test constraint_decider and bt_decider with real track data"""
        track = self.tracks[track_id - 1]
        points = track['left_points'] + track['right_points']
        
        print(f"\n{'='*70}")
        print(f"TESTING TRACK {track_id}")
        print(f"{'='*70}")
        print(f"Left boundary: {len(track['left_points'])} points")
        print(f"Right boundary: {len(track['right_points'])} points")
        
        # Create path pair using first num_points of each boundary
        left_path = list(range(min(num_points, len(track['left_points']))))
        right_path = [i + len(track['left_points']) for i in range(min(num_points, len(track['right_points'])))]
        path_pair = (left_path, right_path)
        
        print(f"\nTesting with:")
        print(f"  Left path: {len(left_path)} points (indices {left_path[:3]}...)")
        print(f"  Right path: {len(right_path)} points (indices {right_path[:3]}...)")
        
        # Test constraint_decider
        print(f"\n--- CONSTRAINT_DECIDER TEST ---")
        try:
            import geo
            geo.points = points
            result = constraint_decider(path_pair)
            print(f"Result: {result}")
            
            # Analyze the result
            self._analyze_constraints(path_pair, points, result)
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        # Test bt_decider
        print(f"\n--- BT_DECIDER TEST ---")
        try:
            bt_result = bt_decider(path_pair)
            print(f"Continue: {bt_result['continue']}")
            print(f"Reason: {bt_result['reason']}")
            print(f"Violations: {len(bt_result['violations'])} types")
            
            for violation_type, violation_data in bt_result['violations']:
                print(f"  {violation_type}: {len(violation_data)} violations")
                if violation_type == 'seg_angle' and violation_data:
                    for side, idx, angle in violation_data[:3]:  # Show first 3
                        print(f"    {side} segment {idx}: {angle:.3f} rad ({np.degrees(angle):.1f}°)")
                        
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_constraints(self, path_pair, points, result):
        """Analyze why constraint_decider returned a specific result"""
        left_path, right_path = path_pair
        
        print(f"\nDetailed Analysis:")
        
        if len(left_path) < 2 or len(right_path) < 2:
            print(f"  Insufficient points for constraint checking")
            return
        
        # Check segment angles
        left_angles = []
        for i in range(len(left_path) - 2):
            p1 = points[left_path[i]]
            p2 = points[left_path[i+1]]
            p3 = points[left_path[i+2]]
            angle = calculate_segment_angle(p1, p2, p3)
            left_angles.append(angle)
        
        if left_angles:
            max_left_angle = max(left_angles)
            print(f"  Left path max angle: {max_left_angle:.3f} rad ({np.degrees(max_left_angle):.1f}°)")
            print(f"    Violates C_seg: {max_left_angle > np.pi/2}")
        
        # Right path angles
        right_angles = []
        for i in range(len(right_path) - 2):
            p1 = points[right_path[i]]
            p2 = points[right_path[i+1]]
            p3 = points[right_path[i+2]]
            angle = calculate_segment_angle(p1, p2, p3)
            right_angles.append(angle)
        
        if right_angles:
            max_right_angle = max(right_angles)
            print(f"  Right path max angle: {max_right_angle:.3f} rad ({np.degrees(max_right_angle):.1f}°)")
            print(f"    Violates C_seg: {max_right_angle > np.pi/2}")
        
        # Check width constraints
        try:
            matching_lines = find_matching_points(left_path, right_path, points)
            widths = [match['width'] for match in matching_lines]
            if widths:
                min_width = min(widths)
                max_width = max(widths)
                mean_width = np.mean(widths)
                print(f"  Width range: {min_width:.3f}m - {max_width:.3f}m (mean: {mean_width:.3f}m)")
                print(f"    Violates C_width: {min_width < 2.5 or max_width > 6.5}")
        except Exception as e:
            print(f"  Width analysis: Failed ({e})")
        
        # Check polygon constraint
        try:
            # Create polygon
            polygon_points = []
            for idx in left_path:
                polygon_points.append(points[idx])
            for idx in reversed(right_path):
                polygon_points.append(points[idx])
            
            # Check for self-intersections
            intersections = self._check_polygon_intersections(polygon_points)
            print(f"  Polygon intersections: {intersections}")
            print(f"    Violates C_poly: {intersections > 0}")
            
        except Exception as e:
            print(f"  Polygon analysis: Failed ({e})")
    
    def _check_polygon_intersections(self, polygon_points):
        """Check for polygon self-intersections"""
        def line_segments_intersect(p1, p2, p3, p4):
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
        
        n = len(polygon_points)
        intersection_count = 0
        
        for i in range(n):
            for j in range(i + 2, n):
                if j == (i + 1) % n or i == (j + 1) % n:
                    continue
                
                if line_segments_intersect(polygon_points[i], polygon_points[(i+1)%n],
                                         polygon_points[j], polygon_points[(j+1)%n]):
                    intersection_count += 1
        
        return intersection_count
    
    def test_all_tracks(self, num_points=10):
        """Test all tracks and provide summary"""
        print(f"\n{'='*70}")
        print("TESTING ALL TRACKS")
        print(f"{'='*70}")
        
        results = []
        for track in self.tracks:
            try:
                self.test_track(track['id'], num_points)
                results.append({
                    'track_id': track['id'],
                    'status': 'completed'
                })
            except Exception as e:
                print(f"\nTrack {track['id']} failed with error: {e}")
                results.append({
                    'track_id': track['id'],
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        successful = sum(1 for r in results if r['status'] == 'completed')
        print(f"Successfully tested: {successful}/{len(results)} tracks")
        
        return results

def main():
    """Main function to run real track tests"""
    print("Real Track Decider Tester")
    print("=" * 70)
    
    tester = RealTrackDeciderTester()
    
    if not tester.tracks:
        print("ERROR: No tracks loaded!")
        return
    
    print(f"Loaded {len(tester.tracks)} tracks")

    for i in range(len(tester.tracks)):
        tester.test_track(i+1, num_points=10)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
