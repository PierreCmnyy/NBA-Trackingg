import cv2
import sys
sys.path.append("../")  # Add parent directory to the path for importing utils
from utils import get_center_of_bbox, get_bbox_width
import numpy as np



def draw_triangle(frame, bbox, color):
    """Draws a triangle on the frame based on the bounding box."""
    y = int(bbox[1])
    x,_ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    # Define triangle points
    triangle_points = np.array([
        [x,y-10],  # Bottom point of the triangle
        [x-10,y-30],  # Left point of the triangle
        [x+10,y-30]   # Right point of the triangle
    ])

    # Draw the triangle
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, color, 2)
    
    return frame



def draw_ellipse(frame, bbox, color, track_id=None):
    """Draws an ellipse on the frame based on the bounding box."""
    y2 = int(bbox[3])
    x_center,_ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)
    cv2.ellipse(frame,                  # put an ellipse on the bottom of the bbox
                (x_center, y2), 
                (int(width),int(0.35*width)), 
                angle = 0, 
                startAngle = -45,
                endAngle = 235,
                color = color, 
                thickness = 2,
                lineType=cv2.LINE_4
                ) 
    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center-rectangle_width//2
    x2_rect = x_center+rectangle_width//2
    y1_rect = (y2-rectangle_height//2)+15
    y2_rect = (y2+rectangle_height//2)+15

    
    if track_id is not None : 
        cv2.rectangle(
            frame,
            (x1_rect, y1_rect),
            (x2_rect, y2_rect),
            color,
            cv2.FILLED,
            )
        
        x1_text = x1_rect + 12
        y1_text = y1_rect + 15
        if track_id > 99 :
            x1_text -= 10 # Adjust text position for larger track IDs

        cv2.putText(
            frame,
            str(track_id),
            (int(x1_text), int(y1_text)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    return frame