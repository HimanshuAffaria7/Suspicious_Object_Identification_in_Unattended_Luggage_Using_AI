# Suspicious Object Detection System

This Python-based system uses deep learning to detect suspicious objects in video footage. It utilizes YOLOv5 for object detection and can process videos using GPU acceleration when available.

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for better performance)
- Required Python packages (listed in requirements.txt)

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your input video in the project directory
2. Update the video path in `detect_objects.py`
3. Run the detection:
   ```bash
   python detect_objects.py
   ```

The script will:
- Process the input video
- Generate an annotated output video
- Print detection results to the console

## Features

- Real-time object detection using YOLOv5
- GPU acceleration support
- Detection of suspicious objects including:
  - Weapons (knives, scissors)
  - Containers (bottles, suitcases, backpacks)
  - Electronic devices (phones, remotes)
- Visualization of detections with bounding boxes
- Confidence scores for each detection
- Timestamp tracking for video analysis

## Output

The system generates:
1. An annotated video showing detections
2. Console output with detailed detection results
3. Classification of objects as 'suspicious' or 'safe'