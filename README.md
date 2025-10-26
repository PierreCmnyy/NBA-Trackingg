# NBA-Analysis

NBA-Analysis is a computer vision project designed to analyze basketball games from video footage. It leverages deep learning models and tracking algorithms to detect and track players and the ball, assign teams, and visualize game dynamics.

## Features
- **Ball and Player Detection:** Uses trained PyTorch models to detect the ball and players in each frame.
- **Tracking:** Implements tracking algorithms to follow player and ball movements throughout the video.
- **Team Assignment:** Assigns players to teams based on their positions and movements.
- **Visualization:** Draws tracks and team control overlays on output videos for analysis.
- **Stub Generation:** Supports stub files for testing and development.

## Project Structure
- `main.py`: Entry point for running the analysis pipeline.
- `ball_acquisition/`: Ball detection and acquisition logic.
- `drawers/`: Visualization utilities for tracks and team control.
- `input_videos/`: Raw basketball game videos for analysis.
- `models/`: Pre-trained PyTorch models for detection.
- `output_videos/`: Processed videos with overlays.
- `stubs/`: Stub files for testing.
- `team_assigner/`: Team assignment logic.
- `trackers/`: Tracking algorithms for ball and players.
- `training_notebooks/`: Jupyter notebooks for model training.
- `utils/`: Utility functions for bounding boxes, stubs, and video processing.

## Getting Started
1. Place your basketball game videos in the `input_videos/` folder.
2. Ensure the required models are present in the `models/` folder.
3. Run `main.py` to start the analysis.
4. Output videos will be saved in `output_videos/`.

## Training
Training notebooks for ball and player detection are available in the `training_notebooks/` folder. You can retrain models or fine-tune them using your own datasets.

## License
This project is for educational and research purposes.

---
Feel free to contribute or open issues for suggestions and improvements.