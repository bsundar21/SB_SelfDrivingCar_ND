"""
Lane Detection Module
Provides functions to detect lanes in frames and videos
Can be imported and used in other scripts
"""

import cv2
import numpy as np
from pathlib import Path
from LaneDetectionAlgorithm import clip_line_to_y_band


def detect_lanes_in_frame(frame):
    """
    Detect lanes in a single frame
    
    Args:
        frame: Input image/frame (BGR format)
        
    Returns:
        Frame with detected lanes drawn (blue for left, red for right)
    """
    # Make a copy to avoid modifying the original
    frame = frame.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define region of interest (ROI)
    height_frame = frame.shape[0]
    width_frame = frame.shape[1]
    
    # ROI polygon (bottom half of frame where lanes are visible)
    polygon = np.array([
        [(0, height_frame), (width_frame, height_frame), 
         (width_frame, height_frame // 2), (0, height_frame // 2)]
    ]) 
    
    # Create mask and apply it
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough line transform to detect lines
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50, 
        minLineLength=50, 
        maxLineGap=10
    )
    
    if lines is not None:
        left_lines = []
        right_lines = []
        
        # Define near-field region (bottom 25% of frame)
        near_field_ratio = 0.25
        y_near_min = int(height_frame * (1 - near_field_ratio))
        y_near_max = height_frame - 1
        
        # Classify lines based on slope
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip vertical lines
            if x1 == x2:
                continue
            
            # Calculate slope
            slope = (y2 - y1) / (x2 - x1)
            
            # Left lane (negative slope, pointing left)
            if slope < -0.5:
                left_lines.append(line[0])
            
            # Right lane (positive slope, pointing right)
            elif slope > 0.5:
                right_lines.append(line[0])
        
        # Draw left lane in blue
        for line in left_lines:
            x1, y1, x2, y2 = line
            clipped = clip_line_to_y_band(
                x1, y1, x2, y2, y_near_min, y_near_max
            )
            if clipped is not None:
                p1, p2 = clipped
                cv2.line(frame, p1, p2, (255, 0, 0), 3)  # Blue
        
        # Draw right lane in red
        for line in right_lines:
            x1, y1, x2, y2 = line
            clipped = clip_line_to_y_band(
                x1, y1, x2, y2, y_near_min, y_near_max
            )
            if clipped is not None:
                p1, p2 = clipped
                cv2.line(frame, p1, p2, (0, 0, 255), 3)  # Red
    
    return frame


def detect_lanes_in_video(input_video, output_video):
    """
    Detect lanes in an entire video and save output
    
    Args:
        input_video: Path to input video file
        output_video: Path to save output video
        
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
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect lanes in frame
            frame_with_lanes = detect_lanes_in_frame(frame)
            
            # Write to output video
            out.write(frame_with_lanes)
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"  ✓ Processed {frame_count} frames...")
    
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
    return True


def extract_lane_info(frame):
    """
    Detect lanes and extract information about them
    
    Args:
        frame: Input image/frame
        
    Returns:
        Dictionary with lane information
    """
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    height = frame.shape[0]
    width = frame.shape[1]
    
    polygon = np.array([
        [(0, height), (width, height), 
         (width, height // 2), (0, height // 2)]
    ]) 
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, 
                           minLineLength=50, maxLineGap=10)
    
    lane_info = {
        'left_lanes': [],
        'right_lanes': [],
        'total_lines': 0,
        'lanes_detected': False
    }
    
    if lines is not None:
        lane_info['total_lines'] = len(lines)
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if x1 == x2:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            if slope < -0.5:
                lane_info['left_lanes'].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'slope': slope})
            elif slope > 0.5:
                lane_info['right_lanes'].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'slope': slope})
        
        lane_info['lanes_detected'] = len(lane_info['left_lanes']) > 0 or len(lane_info['right_lanes']) > 0
    
    return lane_info


if __name__ == "__main__":
    # Example usage
    print("Lane Detector Module")
    print("=" * 60)
    
    # Test with video
    input_video = r"C:\Users\bsund\SB_SelfDrivingCar_ND\data\Video\test.mp4"
    output_video = r"C:\Users\bsund\SB_SelfDrivingCar_ND\data\Video\Output\test_lanes_detected.mp4"
    
    # Uncomment to test:
    # detect_lanes_in_video(input_video, output_video)
    
    print("✓ Lane Detector module ready to use!")