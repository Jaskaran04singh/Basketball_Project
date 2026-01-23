import sys
sys.path.append('../')

from utils.bbox_utils import measure_distance, get_center_of_bbox


class BallAcquisitionDetector:

    def __init__(self):
        self.possession_distance_threshold = 50
        self.min_consecutive_frames = 11
        self.containment_ratio_threshold = 0.8

    def get_player_key_points(self, player_bbox, ball_center):
        ball_x, ball_y = ball_center
        x1, y1, x2, y2 = player_bbox

        width = x2 - x1
        height = y2 - y1

        key_points = []

        if y1 < ball_y < y2:
            key_points.append((x1, ball_y))
            key_points.append((x2, ball_y))

        if x1 < ball_x < x2:
            key_points.append((ball_x, y1))
            key_points.append((ball_x, y2))

        key_points.extend([
            (x1 + width // 2, y1),
            (x2, y1),
            (x1, y1),
            (x2, y1 + height // 2),
            (x1, y1 + height // 2),
            (x1 + width // 2, y1 + height // 2),
            (x2, y2),
            (x1, y2),
            (x1 + width // 2, y2),
            (x1 + width // 2, y1 + height // 3),
        ])

        return key_points

    def compute_ball_containment_ratio(self, player_bbox, ball_bbox):
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        ix1 = max(px1, bx1)
        iy1 = max(py1, by1)
        ix2 = min(px2, bx2)
        iy2 = min(py2, by2)

        if ix2 < ix1 or iy2 < iy1:
            return 0.0

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        ball_area = (bx2 - bx1) * (by2 - by1)

        return intersection_area / ball_area

    def get_min_distance_to_ball(self, ball_center, player_bbox):
        key_points = self.get_player_key_points(player_bbox, ball_center)
        return min(measure_distance(ball_center, point) for point in key_points)

    def select_best_possession_candidate(self, ball_center, players_in_frame, ball_bbox):
        containment_candidates = []
        distance_candidates = []

        for pid, pdata in players_in_frame.items():
            bbox = pdata.get('bbox')
            if not bbox:
                continue

            containment = self.compute_ball_containment_ratio(bbox, ball_bbox)
            distance = self.get_min_distance_to_ball(ball_center, bbox)

            if containment > self.containment_ratio_threshold:
                containment_candidates.append((pid, distance))
            else:
                distance_candidates.append((pid, distance))

        if containment_candidates:
            return max(containment_candidates, key=lambda x: x[1])[0]

        if distance_candidates:
            pid, dist = min(distance_candidates, key=lambda x: x[1])
            if dist < self.possession_distance_threshold:
                return pid

        return -1

    def detect_ball_possession(self, player_tracks, ball_tracks):
        total_frames = len(ball_tracks)
        possession_result = [-1] * total_frames
        possession_streak = {}

        for frame_idx in range(total_frames):
            ball_data = ball_tracks[frame_idx].get(1)
            if not ball_data:
                possession_streak = {}
                continue

            ball_bbox = ball_data.get('bbox')
            if not ball_bbox:
                possession_streak = {}
                continue

            ball_center = get_center_of_bbox(ball_bbox)

            assigned_player = self.select_best_possession_candidate(
                ball_center,
                player_tracks[frame_idx],
                ball_bbox
            )

            if assigned_player != -1:
                possession_streak = {
                    assigned_player: possession_streak.get(assigned_player, 0) + 1
                }

                if possession_streak[assigned_player] >= self.min_consecutive_frames:
                    possession_result[frame_idx] = assigned_player
            else:
                possession_streak = {}

        return possession_result
