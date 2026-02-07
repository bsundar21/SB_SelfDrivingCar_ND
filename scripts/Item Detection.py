from ReadVideo import ReadVideo
import cv2
import numpy as np
from pathlib import Path

# Initialize video reader
video_reader = ReadVideo()

# Try to import YOLO if available, otherwise use OpenCV DNN
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

def init_yolo_model():
    """Initialize YOLO model for car detection"""
    if YOLO_AVAILABLE:
        try:
            model = YOLO('yolov8n.pt')  # nano model for faster inference
            return model
        except Exception as e:
            print(f"Error loading YOLO: {e}")
            return None
    return None

def detect_cars_yolo(frame, model):
    """Detect cars using YOLO"""
    if model is None:
        return frame
    
    results = model(frame)
    
    # Filter detections for car class (class 2 in COCO)
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 2:  # Car class in COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Car {confidence:.2f}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def detect_cars_cascade(frame):
    """Detect cars using OpenCV Haar Cascade (fallback method)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained cascade classifier
    cascade_path = cv2.data.haarcascades + 'haarcascade_car.xml'
    car_cascade = cv2.CascadeClassifier(cascade_path)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1, minSize=(30, 30))
    
    # Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Car', (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def label_cars(frame, model=None):
    """Detect and label cars in a frame"""
    if model is not None:
        return detect_cars_yolo(frame, model)
    else:
        return detect_cars_cascade(frame)

def save_frame(frame, output_dir, frame_num):
    """Save processed frame to file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f"frame_{frame_num:06d}.jpg"
    cv2.imwrite(str(filename), frame)
    return str(filename)

def display_frame(frame, title="Item Detection"):
    """Safely display frame (handles headless environments)"""
    try:
        cv2.imshow(title, frame)
        return True
    except cv2.error as e:
        print(f"Warning: Cannot display frames (headless environment): {e}")
        return False

def main():
    """Main detection pipeline"""
    # Initialize YOLO model
    model = init_yolo_model()
    
    if model is None:
        print("Using Haar Cascade fallback for car detection")
    else:
        print("Using YOLO for car detection")
    
    # Get list of videos if video_reader not initialized with a specific video
    from ReadVideo import get_video_files
    video_files = get_video_files()
    
    if not video_files:
        print("No video files found in data/Video directory")
        return
    
    # Setup output directory
    output_base = Path(r"C:\Users\bsund\SB_SelfDrivingCar_ND\data\Video\Output")
    output_base.mkdir(parents=True, exist_ok=True)
    
    display_enabled = False
    
    for video_file in sorted(video_files):
        print(f"\nProcessing: {Path(video_file).name}")
        
        # Open video
        if not video_reader.open(video_file):
            print(f"  Error: Could not open {video_file}")
            continue
        
        # Get video properties
        fps = video_reader.get_fps()
        width, height = video_reader.get_resolution()
        output_video_path = output_base / f"{Path(video_file).stem}_detected.mp4"
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        frame_num = 0
        
        while True:
            frame = video_reader.read()
            
            if frame is None:
                break
            
            frame_num += 1
            
            # Detect and label cars
            labeled_frame = label_cars(frame, model)
            
            # Write to output video
            out.write(labeled_frame)
            
            # Try to display (only once)
            if not display_enabled and frame_num == 1:
                display_enabled = display_frame(labeled_frame, 'Item Detection')
            
            # Handle keyboard input if display is enabled
            if display_enabled:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_num % 30 == 0:
                print(f"  Processed {frame_num} frames...")
        
        out.release()
        print(f"  Saved output to: {output_video_path}")
    
    # Safe cleanup
    video_reader.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # Ignore GUI errors in headless environments

if __name__ == "__main__":
    main()