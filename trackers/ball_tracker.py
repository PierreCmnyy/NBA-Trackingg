from ultralytics import YOLO
import sys
sys.path.append("../")  # Add parent directory to the path for importing utils
from utils import read_stub, save_stub
import supervision as sv
import numpy as np
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        """Initializes the BallTracker with a YOLO model."""
        self.model = YOLO(model_path)





    def detect_frames(self, frames):
        """Detects players in the given frames using the YOLO model."""
        batch_size=20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size] # Select a subset of 20 frames
            batch_detections = self.model.predict(batch_frames, conf=0.5) # conf=0.5 : minimum confidence threshold of 0.5
            detections += batch_detections
        return detections
    



    def get_object_tracks(self, frames, read_from_stub = False, stub_path=None): 
        """Gets player tracks in the given frames."""

        """ NB : A stub is a file that contains pre-recorded data (in this case, a Python object saved to disk).
        Stubs are used to simulate the result of a costly or unavailable operation (for example, a long computation or an external API), 
        allowing you to test or develop more quickly without having to rerun the entire process each time. """

        tracks = read_stub(read_from_stub, stub_path)  # Read tracks from stub if available
        if tracks is not None:  
            if len(tracks) == len(frames):
                """If tracks are already available and match the number of frames, return them directly."""
                return tracks
            
        
        detections = self.detect_frames(frames)

        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # Get class names mapping (e.g., {0: 'Player'})
            cls_names_inv = {v: k for k, v in cls_names.items()}  # Invert keys and values in the class names dictionary

            detection_supervision = sv.Detections.from_ultralytics(detection)  # Convert YOLO detections to Supervision format
            tracks.append({})
            chosen_bbox = None
            max_confidence = 0
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]

                if cls_id == cls_names_inv["Ball"]:
                    if confidence > max_confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence
                
            if chosen_bbox is not None:
                tracks[frame_num][0] = {"bbox": chosen_bbox}

        save_stub(stub_path, tracks)  # Save the tracks to a stub file for future use

        return tracks  # Return the list of tracking results for all frames
    
    

    def remove_wrong_detections(self,ball_positions):

        maximum_allowed_distance = 15
        last_good_frame_index= -1

        for i in range(len(ball_positions)):
            current_bbox = ball_positions[i].get(0,{}).get('bbox', [])

            if len(current_bbox) == 0:
                continue

            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(0,{}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_maximum_distance = maximum_allowed_distance * frame_gap # the maximum allowed distance increases with the number of frames since the ball can move further in each frame

     # calculate the distance between the current and last good bounding boxes
            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_bbox[:2])) > adjusted_maximum_distance:
                # If the distance is too large, remove the current detection
                ball_positions[i] = {}
            else:
                # If the distance is acceptable, update the last good frame index
                last_good_frame_index = i

        return ball_positions  # Return the cleaned list of ball positions after removing wrong detections
    

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [ x.get(0,{}).get('bbox',[]) for x in ball_positions ]  #
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values in the DataFrame
        df_ball_positions = df_ball_positions.interpolate() # You can see the Notebook tmp to see what .interpolate() and .bfill() do
        df_ball_positions = df_ball_positions.bfill() 

        ball_positions = [{0:{"bbox":x}} for x in df_ball_positions.to_numpy().tolist()] # Giving the format expected by the tracker

        return ball_positions  