import cv2
import os
from pathlib import Path

class ReadVideo:
    """Class to read video frames from a file or directory"""
    
    def __init__(self, video_path=None):
        """
        Initialize video reader
        
        Args:
            video_path: Path to video file (optional)
        """
        self.video_path = video_path
        self.cap = None
        self.current_frame = None
        self.frame_count = 0
        
        if video_path and os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                self.cap = None
    
    def read(self):
        """Read next frame from video"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            self.current_frame = frame
            return frame
        else:
            return None
    
    def open(self, video_path):
        """Open a new video file"""
        if self.cap:
            self.cap.release()
        
        if os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
            self.frame_count = 0
            self.video_path = video_path
            return self.cap.isOpened()
        return False
    
    def release(self):
        """Release the video capture"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_fps(self):
        """Get frames per second of video"""
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FPS))
        return 0
    
    def get_frame_count(self):
        """Get total frame count"""
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0
    
    def get_resolution(self):
        """Get video resolution as (width, height)"""
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return None
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.release()


# Utility functions
INPUT_DIR = r"C:\Users\bsund\SB_SelfDrivingCar_ND\data\Video"
OUTPUT_DIR = r"C:\Users\bsund\SB_SelfDrivingCar_ND\data\Video\Output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get list of video files
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

def get_video_files():
    """Returns list of video files from INPUT_DIR"""
    video_files = []
    if os.path.exists(INPUT_DIR):
        video_files = [
            os.path.join(INPUT_DIR, f) 
            for f in os.listdir(INPUT_DIR) 
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
        ]
    return video_files

def get_output_path(video_filename):
    """Returns output path for a given video filename"""
    return os.path.join(OUTPUT_DIR, video_filename)