import cv2
import numpy as np
from pathlib import Path

def clip_line_to_y_band(x1, y1, x2, y2, y_min, y_max):
    """Clip a line segment so only the part inside [y_min, y_max] remains."""
    points = []

    # Keep endpoints already inside the near-field band
    if y_min <= y1 <= y_max:
        points.append((x1, y1))
    if y_min <= y2 <= y_max:
        points.append((x2, y2))

    # Add intersections with band boundaries
    if y1 != y2:
        for yb in (y_min, y_max):
            if min(y1, y2) <= yb <= max(y1, y2):
                t = (yb - y1) / (y2 - y1)
                xb = int(x1 + t * (x2 - x1))
                points.append((xb, yb))

    # Deduplicate and return two points if possible
    unique = []
    for p in points:
        if p not in unique:
            unique.append(p)

    if len(unique) < 2:
        return None

    # Pick the farthest two points to form the clipped segment
    max_dist = -1
    best_pair = None
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            dx = unique[i][0] - unique[j][0]
            dy = unique[i][1] - unique[j][1]
            dist = dx * dx + dy * dy
            if dist > max_dist:
                max_dist = dist
                best_pair = (unique[i], unique[j])
    return best_pair

def detect_lanes(video_path, output_path):
    """Detect lanes in video and save output with marked lanes."""
    
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    video_path = Path(video_path)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping unreadable video: {video_path}")
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    output_file = Path(output_path) / f"{video_path.stem}_lanes_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Define region of interest (ROI)
        height_frame = frame.shape[0]
        width_frame = frame.shape[1]
        polygon = np.array([
            [(0, height_frame), (width_frame, height_frame), 
             (width_frame, height_frame // 2), (0, height_frame // 2)]
        ]) 
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough line transform
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            left_lines = []
            right_lines = []
            near_field_ratio = 0.25  # Bottom 25% of frame ~= immediate area in front
            y_near_min = int(height_frame * (1 - near_field_ratio))
            y_near_max = height_frame - 1
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                
                # Left lane (negative slope)
                if slope < -0.5:
                    left_lines.append(line[0])
                # Right lane (positive slope)
                elif slope > 0.5:
                    right_lines.append(line[0])
            
            # Draw left lane in blue
            for line in left_lines:
                clipped = clip_line_to_y_band(
                    line[0], line[1], line[2], line[3], y_near_min, y_near_max
                )
                if clipped is not None:
                    p1, p2 = clipped
                    cv2.line(frame, p1, p2, (255, 0, 0), 3)
            
            # Draw right lane in red
            for line in right_lines:
                clipped = clip_line_to_y_band(
                    line[0], line[1], line[2], line[3], y_near_min, y_near_max
                )
                if clipped is not None:
                    p1, p2 = clipped
                    cv2.line(frame, p1, p2, (0, 0, 255), 3)
        
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Output saved to {output_file}")

# Main execution
video_input = r"C:\Users\bsund\SB_SelfDrivingCar_ND\data\Video"
video_output = r"C:\Users\bsund\SB_SelfDrivingCar_ND\data\Video\Output"

# Process all video files in the input folder
video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
video_files = [
    p for p in Path(video_input).iterdir()
    if p.is_file() and p.suffix.lower() in video_extensions
]

if not video_files:
    print(f"No video files found in {video_input}")
else:
    for video_file in sorted(video_files):
        print(f"Processing {video_file.name}...")
        detect_lanes(str(video_file), video_output)
