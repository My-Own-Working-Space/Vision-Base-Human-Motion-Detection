import os
import sys
import unittest
import numpy as np
import cv2
from PIL import Image
import torch

# Add paths to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inference')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))

from yolo_detector import YOLODetector
from resnet_lstm_classifier import ResNetLSTMClassifier

class TestFunctionalAI(unittest.TestCase):
    """
    Functional tests for the core AI pipeline components (YOLO, ByteTrack, ResNet, LSTM).
    """

    @classmethod
    def setUpClass(cls):
        # Locate the test image from ultralytics assets
        cls.test_image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '../.venv/lib/python3.12/site-packages/ultralytics/assets/zidane.jpg'
        ))
        if not os.path.exists(cls.test_image_path):
            # Fallback if not found: create a random noise image
            cls.test_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_sample.jpg'))
            dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(cls.test_image_path, dummy_img)
            cls.is_fallback = True
        else:
            cls.is_fallback = False

        cls.yolo_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/yolov8n.pt'))
        cls.lstm_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/resnet_lstm_best.pth'))

        # Initialize detector & classifier
        cls.detector = YOLODetector(cls.yolo_model_path)
        cls.classifier = ResNetLSTMClassifier(cls.lstm_model_path)

    def test_yolo_detection_and_tracking(self):
        """Test that YOLODetector correctly detects persons and returns bounding box crops."""
        frame = cv2.imread(self.test_image_path)
        self.assertIsNotNone(frame, "Failed to load test image.")

        # Run detection using the legacy detect method first
        results_detect = self.detector.detect(frame)
        self.assertIsNotNone(results_detect, "YOLO detector returned None results.")
        crops_detect = self.detector.get_crops(frame, results_detect)
        
        if not self.is_fallback:
            # Zidane.jpg has persons
            self.assertGreater(len(crops_detect), 0, "No persons detected in test image.")
            self.assertEqual(crops_detect[0]['label'], 'person', "First detection class should be 'person'.")

        # Run ByteTrack tracking
        results_track = self.detector.track(frame)
        self.assertIsNotNone(results_track, "YOLO tracker returned None results.")
        
        crops_track = self.detector.get_tracked_crops(frame, results_track)
        # Tracking requires multiple frames to assign track ID consistently, but we can verify structure
        for crop in crops_track:
            self.assertIn('image', crop)
            self.assertIsInstance(crop['image'], Image.Image)
            self.assertIn('bbox', crop)
            self.assertEqual(len(crop['bbox']), 4)
            self.assertIn('track_id', crop)
            self.assertIsInstance(crop['track_id'], int)
            self.assertIn('conf', crop)
            self.assertIsInstance(crop['conf'], float)

    def test_resnet_feature_extraction(self):
        """Test that the ResNet18 backbone correctly extracts 512-dimensional features."""
        # Create a dummy PIL image crop
        dummy_crop = Image.fromarray(np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8))
        feat = self.classifier._extract_feature(dummy_crop)
        
        self.assertIsInstance(feat, torch.Tensor)
        self.assertEqual(feat.ndim, 1)
        self.assertEqual(feat.shape[0], 512, "ResNet feature should be 512-dimensional.")

    def test_resnet_lstm_temporal_prediction_flow(self):
        """Test the sliding window buffer flow and behavior classification."""
        dummy_crop = Image.fromarray(np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8))
        track_id = 999

        # Clean any old buffers
        if track_id in self.classifier.track_buffers:
            del self.classifier.track_buffers[track_id]

        # Feed 15 frames: should return 'Buffering...'
        for i in range(1, 16):
            label, confidence, buf_len = self.classifier.predict(dummy_crop, track_id)
            self.assertEqual(label, 'Buffering...', f"Should be buffering at frame {i}")
            self.assertEqual(confidence, 0.0)
            self.assertEqual(buf_len, i)

        # Feed the 16th frame: should perform temporal inference and return either 'Normal' or 'Anomaly'
        label, confidence, buf_len = self.classifier.predict(dummy_crop, track_id)
        self.assertIn(label, ['Normal', 'Anomaly'], "Prediction label must be 'Normal' or 'Anomaly'.")
        self.assertGreater(confidence, 0.0, "Confidence should be a positive float after inference.")
        self.assertEqual(buf_len, 16, "Buffer length should remain capped at 16.")

        # Test buffer cleanup
        self.assertIn(track_id, self.classifier.track_buffers)
        # Exceed grace period to trigger deletion
        for _ in range(self.classifier.grace_period + 5):
            self.classifier.cleanup_tracks(active_ids={1, 2, 3})
        self.assertNotIn(track_id, self.classifier.track_buffers, "Stale track buffer was not cleared.")

if __name__ == '__main__':
    unittest.main()
