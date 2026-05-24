import os
import sys
import cv2
import yaml
import time
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inference')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))

from yolo_detector import YOLODetector
from resnet_lstm_classifier import ResNetLSTMClassifier

VIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inference', 'test_video.mp4'))
BYTETRACK_CFG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inference', 'bytetrack_custom.yaml'))

def update_bytetrack_config(track_buffer_val):
    """Dynamically updates the track_buffer value in bytetrack_custom.yaml."""
    if not os.path.exists(BYTETRACK_CFG):
        print(f"Error: {BYTETRACK_CFG} not found!")
        return
        
    with open(BYTETRACK_CFG, 'r') as f:
        data = yaml.safe_load(f)
        
    data['track_buffer'] = track_buffer_val
    
    with open(BYTETRACK_CFG, 'w') as f:
        yaml.safe_dump(data, f)
    print(f"\n[Config] Updated track_buffer to {track_buffer_val} in config file.")

def run_diagnostic_stream(track_buffer_val, num_frames=60, yolo_conf=0.15):
    update_bytetrack_config(track_buffer_val)
    
    yolo_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/yolov8n.pt'))
    detector = YOLODetector(yolo_model_path, device_mode='server')
    # Set YOLO conf threshold dynamically
    detector.model.overrides['conf'] = yolo_conf
    
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'resnet_lstm_best.pth'))
    classifier = ResNetLSTMClassifier(model_path)
    classifier.grace_period = 30 # standard user request
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open test video at {VIDEO_PATH}")
        return {}
        
    frame_idx = 0
    unique_ids = set()
    track_history = {} # track_id -> list of frames it appeared in
    
    print(f"\n--- Running Diagnostics (track_buffer={track_buffer_val}, yolo_conf={yolo_conf}) ---")
    
    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Resize for consistent speed
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
            
        # YOLO + ByteTrack
        results = detector.track(frame)
        crops = detector.get_tracked_crops(frame, results)
        
        active_ids = set()
        print(f"Frame #{frame_idx:02d}: Detected {len(crops)} person(s)")
        
        for crop in crops:
            tid = crop['track_id']
            active_ids.add(tid)
            unique_ids.add(tid)
            
            if tid not in track_history:
                track_history[tid] = []
            track_history[tid].append(frame_idx)
            
            label, conf, buf_len = classifier.predict(crop['image'], tid)
            
        classifier.cleanup_tracks(active_ids)
        
    cap.release()
    
    # Calculate stats
    print("\n--- Diagnostic Results ---")
    print(f"Total processed frames: {frame_idx}")
    print(f"Unique Track IDs assigned: {len(unique_ids)}")
    print(f"List of unique Track IDs: {sorted(list(unique_ids))}")
    for tid, frames in track_history.items():
        print(f"  Track ID {tid}: Appeared in {len(frames)} frames. Range: [{min(frames)} - {max(frames)}]")
        
    return {
        'unique_ids_count': len(unique_ids),
        'track_history': track_history
    }

if __name__ == '__main__':
    # Run with different track_buffer configurations as requested
    results_30 = run_diagnostic_stream(track_buffer_val=30, num_frames=60, yolo_conf=0.15)
    results_60 = run_diagnostic_stream(track_buffer_val=60, num_frames=60, yolo_conf=0.15)
    results_120 = run_diagnostic_stream(track_buffer_val=120, num_frames=60, yolo_conf=0.15)
