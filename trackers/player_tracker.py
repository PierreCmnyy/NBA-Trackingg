from ultralytics import YOLO
import supervision as sv
import sys 
sys.path.append("../") # in order to go back to the parent directory and import utils
from utils import read_stub, save_stub

class PlayerTracker:
    """Encapsultes all the logic in one class so that the code can be structured and well maintained."""

    def __init__(self, model_path):
        """Initializes the PlayerTracker with a YOLO model."""
        self.model = YOLO(model_path) # YOLO = "who is where on a frame"
        self.tracker = sv.ByteTrack() # ByteTrack = "who is who among the frames"

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

        """If tracks are not available, perform detection, tracking and put it into a stub."""
        detections = self.detect_frames(frames)  # Run detection on all frames
        tracks = []  # List to store tracking results for each frame

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # Get class names mapping (e.g., {0: 'Player'})
            cls_names_inv = {v: k for k, v in cls_names.items()}  # Invert keys and values in the class names dictionary

            detection_supervision = sv.Detections.from_ultralytics(detection)  # Convert YOLO detections to Supervision format

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)  # Update tracker and get tracked objects

            tracks.append({})  # Initialize an empty dict for this frame
        
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
            
                if cls_id == cls_names_inv["Player"]:  # Only keep detections for the 'Player' class
                    tracks[frame_num][track_id] = {"bbox" : bbox}  # Store bounding box for this player and frame

        save_stub(stub_path, tracks)  # Save the tracks to a stub file for future use

        return tracks  # Return the list of tracking results for all frames