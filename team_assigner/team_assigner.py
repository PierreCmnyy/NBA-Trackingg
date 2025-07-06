from PIL import Image
from utils import read_stub
from transformers import CLIPProcessor, CLIPModel
import cv2


class TeamAssigner:
    def __init__(self,team_1_class_name="white shirt", team_2_class_name="white shirt"):

        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        self.player_team_dict = {}  # Dictionary to store player IDs and their assigned teams
    def load_model(self) : 
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] #To crop the image to the player bbox
        
        #Converting from bgr to rgb
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        classes = [self.team_1_class_name,self.team_2_class_name]

        inputs = self.processor(text=classes, images=pil_image, return_tensors="pt", padding=True)  

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax

        class_name = classes[probs.argmax(dim=1)[0]]
        return class_name

    def get_player_team(self, frame, player_bbox, player_id):
        """Assigns a team to a player based on their bounding box."""

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = 2
        if player_color == self.team_1_class_name:
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id 
    
    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        """   Assigns teams to players across multiple frames based on their bounding boxes. """

        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment
            return player_assignment
        
        self.load_model()

        player_assignment = []

        for frame_num,player_track in enumerate(player_tracks):
            player_assignment.append({})
            for player_id, player_bbox in player_track.items():
                team = self.get_player_team(video_frames[frame_num],player_bbox=player_bbox['bbox'], player_id=player_id)
                player_assignment[frame_num][player_id] = team

        return player_assignment
    
