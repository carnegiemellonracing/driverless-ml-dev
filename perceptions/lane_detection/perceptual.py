from typing import List

def nearest_point(point, boundary):
    pass

class PerceptualField:
    def __init__(self, heading: float, x: float, y: float) -> None:
        pass

def generate_perceptual_fields(adj_graph: dict, left_bound: List[int], right_bound: List[int]) -> List[PerceptualField]:
    perceptual_field_data = []
    for left_point in left_bound:
        right_point = nearest_point(left_point, right_bound)
        heading = get_angle(left_point, right_point)


        