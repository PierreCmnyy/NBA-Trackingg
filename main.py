
import cv2
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import  PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer

from ball_acquisition import BallAcquisitionDetector
from team_assigner import TeamAssigner
def main():

    # Read video
    video_frames = read_video("input_videos/video_1.mp4")


    # Initialize Tracker
    player_tracker = PlayerTracker("models/player_detector.pt") 
    ball_tracker = BallTracker("models/ball_detector_model.pt")

    # Run trackers
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="stubs/player_track_stubs.pkl")

    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="stubs/ball_track_stubs.pkl")
    
    # Remove wrong ball detections

    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)

    #Interpolate ball positions

    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)


    # Assign teams to players
    team_assigner = TeamAssigner() 
    player_teams = team_assigner.get_player_teams_across_frames(
        video_frames,
        player_tracks,
        read_from_stub=True,
        stub_path="stubs/player_assjgnment_stubs.pkl")
    
    print(player_teams)
        
    # Ball acquisition
    ball_acquisition_detector = BallAcquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks, ball_tracks)
    print(ball_acquisition)

    
    # Draw Output 
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()

    # Draw player tracks on video frames
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks,ball_acquisition, player_teams)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_teams, ball_acquisition)

    # Save Video
    save_video(output_video_frames, "output_videos/output_video.avi")



    # Nombre de frames dans la vidéo d'entrée
    cap_in = cv2.VideoCapture("input_videos/video_1.mp4")
    print("Nombre de frames dans video_1 :", int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap_in.release()

    # Nombre de frames dans la vidéo de sortie
    cap_out = cv2.VideoCapture("output_videos/output_video.avi")
    print("Nombre de frames dans output_video :", int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap_out.release()
    
if __name__ == "__main__":
    main()