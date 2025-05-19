import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

class SuspiciousObjectDetector:
    def __init__(self):
        # Initialize CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to(self.device)
        
        # Define suspicious objects
        self.suspicious_objects = [
            'knife', 'scissors', 'bottle', 'cell phone', 'remote',
            'suitcase', 'backpack', 'handbag'
        ]
        
    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video file")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path is specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        results = []
        frame_count = 0
        
        # Process every 2 seconds of video
        frame_interval = fps * 2
        
        # Create progress bar
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        # Create autocast context based on device
        autocast_context = torch.amp.autocast(device_type=self.device.type)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process frames at the specified interval
            if frame_count % frame_interval == 0:
                # Process frame using the new autocast syntax
                with autocast_context:
                    frame_results = self.detect_objects(frame, frame_count / fps)
                    results.extend(frame_results)
                
                # Draw detections on frame
                annotated_frame = self.draw_detections(frame, frame_results)
                
                # Write frame if output path is specified
                if output_path:
                    out.write(annotated_frame)
            
            # Update progress bar
            pbar.update(1)
            frame_count += 1
        
        # Clean up
        pbar.close()
        cap.release()
        if output_path:
            out.release()
        
        return results
    
    def detect_objects(self, frame, timestamp):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(frame_rgb)
        
        # Process detections
        detections = []
        for *box, conf, cls in results.xyxy[0]:  # xyxy format
            obj_class = results.names[int(cls)]
            
            detection = {
                'timestamp': timestamp,
                'confidence': float(conf),
                'status': 'suspicious' if obj_class.lower() in self.suspicious_objects else 'safe',
                'object': obj_class,
                'bbox': [float(x) for x in box]
            }
            detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame, detections):
        frame_copy = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Set color based on status
            color = (0, 0, 255) if detection['status'] == 'suspicious' else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{detection['object']} ({detection['confidence']:.2f})"
            cv2.putText(frame_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_copy

def main():
    # Initialize detector
    detector = SuspiciousObjectDetector()
    
    # Process video - use raw string for path
    video_path = r"input_video.mp4"  # Replace with your video path
    output_path = r"output_video.mp4"
    
    print("Starting video processing...")
    results = detector.process_video(video_path, output_path)
    
    # Print results
    print("\nDetection Results:")
    for result in results:
        print(f"Time: {result['timestamp']:.2f}s - "
              f"Found {result['object']} "
              f"(Confidence: {result['confidence']:.2f}) - "
              f"Status: {result['status']}")

if __name__ == "__main__":
    main()