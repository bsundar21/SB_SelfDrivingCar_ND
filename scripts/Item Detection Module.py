"""
Item Detection Module
Provides functions to detect items (cars, vehicles) in frames and videos
Can be imported and used in other scripts
Supports both YOLO and Haar Cascade detection methods
"""

import cv2
import numpy as np
from pathlib import Path

# Try to import YOLO if available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ==================== GLOBAL YOLO MODEL ====================
_yolo_model = None


def init_yolo_model():
    """
    Initialize YOLO model (loads once and reuses)
    
    Returns:
        YOLO model instance or None if not available
    """
    global _yolo_model
    
    if _yolo_model is not None:
        return _yolo_model
    
    if YOLO_AVAILABLE:
        try:
            print("  Loading YOLO model...")
            _yolo_model = YOLO('yolov8n.pt')  # nano model for faster inference
            print("  ✓ YOLO model loaded successfully")
            return _yolo_model
        except Exception as e:
            print(f"  ⚠️  Error loading YOLO: {e}")
            return None
    else:
        print("  ⚠️  YOLO not installed. Install with: pip install ultralytics")
    
    return None


# ==================== ITEM DETECTION FUNCTIONS ====================

def detect_items_yolo_in_frame(frame, model=None, confidence_threshold=0.5):
    """
    Detect items (cars) using YOLO in a single frame
    
    Args:
        frame: Input image/frame (BGR format)
        model: YOLO model instance (optional, will initialize if None)
        confidence_threshold: Minimum confidence score for detection
        
    Returns:
        Frame with detected items drawn, and list of detections
    """
    if model is None:
        model = init_yolo_model()
    
    detection_results = []
    
    if model is None:
        return frame, detection_results
    
    try:
        results = model(frame, conf=confidence_threshold, verbose=False)
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                
                # Class 2 = Car, 5 = Bus, 7 = Truck in COCO dataset
                # You can add more vehicle classes as needed
                if class_id in [2, 5, 7]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf)
                    
                    # Get class name
                    class_names = {2: 'Car', 5: 'Bus', 7: 'Truck'}
                    class_name = class_names.get(class_id, 'Vehicle')
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Store detection info
                    detection_results.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'center_x': (x1 + x2) // 2,
                        'center_y': (y1 + y2) // 2
                    })
    
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
    
    return frame, detection_results


def detect_items_cascade_in_frame(frame):
    """
    Detect items (cars) using Haar Cascade in a single frame
    
    Args:
        frame: Input image/frame (BGR format)
        
    Returns:
        Frame with detected items drawn, and list of detections
    """
    detection_results = []
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load pre-trained cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_car.xml'
        car_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Detect cars
        cars = car_cascade.detectMultiScale(gray, 1.1, 1, minSize=(30, 30))
        
        # Draw rectangles around detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Car', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Store detection info
            detection_results.append({
                'class': 'Car',
                'confidence': 0.9,  # Cascade doesn't provide confidence
                'bbox': (x, y, x + w, y + h),
                'center_x': x + w // 2,
                'center_y': y + h // 2
            })
    
    except Exception as e:
        print(f"Error in Cascade detection: {e}")
    
    return frame, detection_results


def detect_items_in_frame(frame, use_yolo=True, confidence_threshold=0.5):
    """
    Detect items in a single frame (automatic method selection)
    
    Args:
        frame: Input image/frame (BGR format)
        use_yolo: If True, try YOLO first; if False, use Cascade
        confidence_threshold: Minimum confidence for YOLO detections
        
    Returns:
        Frame with detected items drawn, and list of detections
    """
    if use_yolo:
        model = init_yolo_model()
        if model is not None:
            return detect_items_yolo_in_frame(frame, model, confidence_threshold)
    
    return detect_items_cascade_in_frame(frame)


def detect_items_in_video(input_video, output_video, use_yolo=True, confidence_threshold=0.5):
    """
    Detect items in an entire video and save output
    
    Args:
        input_video: Path to input video file
        output_video: Path to save output video
        use_yolo: If True, use YOLO; if False, use Cascade
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        True if successful, False otherwise
    """
    input_path = Path(input_video)
    
    # Validate input
    if not input_path.exists():
        print(f"❌ Error: Input video not found: {input_video}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {input_video}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing: {input_path.name}")
    print(f"  Resolution: {width}x{height} @ {fps} FPS")
    
    if use_yolo:
        print("  Using: YOLO detection")
    else:
        print("  Using: Cascade detection")
    
    # Create output directory
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Error: Cannot create output video")
        cap.release()
        return False
    
    frame_count = 0
    total_items_detected = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect items in frame
            frame_with_items, detections = detect_items_in_frame(
                frame, use_yolo=use_yolo, confidence_threshold=confidence_threshold
            )
            
            total_items_detected += len(detections)
            
            # Write to output video
            out.write(frame_with_items)
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"  ✓ Processed {frame_count} frames ({total_items_detected} items detected)...")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        cap.release()
        out.release()
        return False
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"✅ Output saved to: {output_path}")
    print(f"   Total frames: {frame_count}")
    print(f"   Total items detected: {total_items_detected}")
    return True


def extract_item_info(frame, use_yolo=True, confidence_threshold=0.5):
    """
    Detect items and extract detailed information
    
    Args:
        frame: Input image/frame
        use_yolo: Use YOLO if True, Cascade if False
        confidence_threshold: Minimum confidence for YOLO detections
        
    Returns:
        Dictionary with item detection information
    """
    frame_copy = frame.copy()
    frame_with_items, detections = detect_items_in_frame(
        frame_copy, use_yolo=use_yolo, confidence_threshold=confidence_threshold
    )
    
    item_info = {
        'total_items': len(detections),
        'detections': detections,
        'frame': frame_with_items,
        'cars': [d for d in detections if d['class'] == 'Car'],
        'buses': [d for d in detections if d['class'] == 'Bus'],
        'trucks': [d for d in detections if d['class'] == 'Truck']
    }
    
    return item_info


def count_items_in_video(input_video, use_yolo=True, confidence_threshold=0.5):
    """
    Count items detected in a video without saving output
    Useful for analysis
    
    Args:
        input_video: Path to input video file
        use_yolo: Use YOLO if True, Cascade if False
        confidence_threshold: Minimum confidence for YOLO detections
        
    Returns:
        Dictionary with detection statistics
    """
    input_path = Path(input_video)
    
    if not input_path.exists():
        print(f"❌ Error: Input video not found: {input_video}")
        return None
    
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video")
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    total_detections = 0
    max_detections_per_frame = 0
    frames_with_detections = 0
    
    print(f"Analyzing: {input_path.name}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            _, detections = detect_items_in_frame(frame, use_yolo=use_yolo)
            
            if len(detections) > 0:
                frames_with_detections += 1
                total_detections += len(detections)
                max_detections_per_frame = max(max_detections_per_frame, len(detections))
            
            if frame_count % 30 == 0:
                print(f"  Analyzed {frame_count} frames...")
    
    finally:
        cap.release()
    
    stats = {
        'video_file': input_path.name,
        'total_frames': frame_count,
        'fps': fps,
        'duration_seconds': frame_count / fps if fps > 0 else 0,
        'frames_with_detections': frames_with_detections,
        'total_detections': total_detections,
        'avg_detections_per_frame': total_detections / frame_count if frame_count > 0 else 0,
        'max_detections_per_frame': max_detections_per_frame
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("Item Detection Module")
    print("=" * 60)
    
    # Test with video
    input_video = r"C:\Users\bsund\SB_SelfDrivingCar_ND\data\Video\test.mp4"
    output_video = r"C:\Users\bsund\SB_SelfDrivingCar_ND\data\Video\Output\test_items_detected.mp4"
    
    # Uncomment to test:
    # detect_items_in_video(input_video, output_video, use_yolo=True)
    
    # Uncomment to analyze video:
    # stats = count_items_in_video(input_video, use_yolo=True)
    # if stats:
    #     print(f"\nStatistics:")
    #     for key, value in stats.items():
    #         print(f"  {key}: {value}")
    
    print("✓ Item Detection module ready to use!")