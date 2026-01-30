from utils.video_utils import read_video, save_video
from trackers import PlayerTracker
def main():
    
    #read input video
    video_frames = read_video("input/video_1.mp4")
    
    #initialise player tracker
    model_path = "models/player_detector.pt"
    player_tracker = PlayerTracker(model_path)
    
    #run player tracking
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="stubs/player_track_stubs.pkl"
                                                    )
    print(player_tracks)
    
    #Save processed video
    save_video(video_frames, "output/output_video_1.avi")
    
if __name__ == "__main__":
    main()
    