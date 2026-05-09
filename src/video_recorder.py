import cv2
import os
from datetime import datetime

def get_video_writer(output_dir, frame_width, frame_height, fps=20):
    os.makedirs(output_dir, exist_ok=True)
    filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.mp4")
    path = os.path.join(output_dir, filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (frame_width, frame_height))
    return writer, path