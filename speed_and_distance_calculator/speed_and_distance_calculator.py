import os
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import measure_distance


class SpeedAndDistanceCalculator():
    def __init__(self,width_in_pixels,height_in_pixels,width_in_meters,height_in_meters):
        
        self.width_in_pixels = width_in_pixels
        self.height_in_pixels= height_in_pixels
        self.width_in_meters = width_in_meters
        self.height_in_meters= height_in_meters

    def calculate_distance(self,tactical_player_positions):
        previous_players_position = {}
        output_distances =[]

        for frame_number, tactical_player_position_frame in enumerate(tactical_player_positions):
            output_distances.append({})

            for player_id, current_player_position in tactical_player_position_frame.items():
                if player_id in previous_players_position:
                    previous_position = previous_players_position[player_id]
                    meter_distance = self.calculate_meter_distance(previous_position, current_player_position)
                    output_distances[frame_number][player_id] = meter_distance

                previous_players_position[player_id]=current_player_position    
        return output_distances

    def calculate_meter_distance(self,previous_pixel_position, current_pixel_position):
         previous_pixel_x, previous_pixel_y = previous_pixel_position
         current_pixel_x, current_pixel_y = current_pixel_position
         previous_meter_x = previous_pixel_x * self.width_in_meters / self.width_in_pixels
         previous_meter_y = previous_pixel_y * self.height_in_meters / self.height_in_pixels
         current_meter_x = current_pixel_x * self.width_in_meters / self.width_in_pixels
         current_meter_y = current_pixel_y * self.height_in_meters / self.height_in_pixels
         meter_distance =measure_distance((current_meter_x,current_meter_y),
                                          (previous_meter_x,previous_meter_y)
                                          )
         meter_distance = meter_distance*0.4
         return meter_distance
    
    def calculate_speed(self, distances, fps=30):
        speeds = []
        window_size = 5 
        for frame_idx in range(len(distances)):
            speeds.append({})
            for player_id in distances[frame_idx].keys():
                start_frame = max(0, frame_idx - (window_size * 3) + 1)
                total_distance = 0
                frames_present = 0
                last_frame_present = None
                for i in range(start_frame, frame_idx + 1):
                    if player_id in distances[i]:
                        if last_frame_present is not None:
                            total_distance += distances[i][player_id]
                            frames_present += 1
                        last_frame_present = i
       
                if frames_present >= window_size:
                    time_in_seconds = frames_present / fps
                    time_in_hours = time_in_seconds / 3600
                    if time_in_hours > 0:
                        speed_kmh = (total_distance / 1000) / time_in_hours
                        speeds[frame_idx][player_id] = speed_kmh
                    else:
                        speeds[frame_idx][player_id] = 0
                else:
                    speeds[frame_idx][player_id] = 0
        return speeds