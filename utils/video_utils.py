import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True :
        ret, frame = cap.read() #return if something is returned or not and the frame
        if not ret: #it means that the video is finished and we can break
            break

        frames.append(frame) #otherwise we want to append the frame to the list
    return frames

def save_video(output_video_frames, output_video_path) : 
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.mkdir(os.path.dirname(output_video_path))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    out = cv2.VideoWriter(output_video_path,fourcc, 24.0, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)  # Write each frame to the video file
    out.release()  # Release the video writer object

