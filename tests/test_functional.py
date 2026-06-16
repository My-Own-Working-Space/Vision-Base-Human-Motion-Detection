"""
Functional tests for the refactored event-driven edge device architecture.
"""

import os
import sys
import unittest
import numpy as np
import cv2
import shutil
import csv
from datetime import datetime
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from edge.config import load_config
from edge.models import DetectionEvent, EventType, AlertPayload
from edge.services.camera_service import CameraService
from edge.services.detection_service import DetectionService
from edge.services.alert_service import AlertService
from edge.clients.api_client import ApiClient


class TestFunctionalEdge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.test_image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '../.venv/lib/python3.12/site-packages/ultralytics/assets/zidane.jpg'
        ))
        if not os.path.exists(cls.test_image_path):
            cls.test_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_sample.jpg'))
            dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(cls.test_image_path, dummy_img)

    def test_camera_service_state_transitions(self):
        """Test CameraService initialization and state management."""
        camera_service = CameraService(
            camera_config=self.config.camera,
            state_config=self.config.state_machine,
        )
        self.assertEqual(camera_service.state, "IS_PROCESSING")
        
        # Feed frame to trigger state evaluation
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = camera_service.update_state(dummy_frame)
        self.assertIn(state, ["IS_PROCESSING", "SLEEPING"])

    def test_detection_service_flow(self):
        """Test DetectionService processing a frame."""
        detection_service = DetectionService(
            detection_config=self.config.detection,
            state_machine_config=self.config.state_machine,
        )
        detection_service.initialize()
        
        frame = cv2.imread(self.test_image_path)
        results = detection_service.process_frame(frame)
        
        # Results should be a list of (DetectionEvent, evidence_frame_or_None)
        self.assertIsInstance(results, list)
        for event, evidence in results:
            self.assertIsInstance(event, DetectionEvent)
            self.assertIn(event.event_type, [EventType.NORMAL, EventType.ANOMALY, EventType.BUFFERING])
            self.assertIsInstance(event.confidence, float)
            self.assertIsInstance(event.track_id, int)

    def test_alert_service_evaluation_and_csv(self):
        """Test AlertService threshold check, CSV logging, and local file storage."""
        # Setup clean alerts directory
        alerts_dir = "test_alerts_tmp"
        if os.path.exists(alerts_dir):
            shutil.rmtree(alerts_dir)
        
        # Override alert config for test
        from edge.config import AlertConfig, ApiConfig
        alert_cfg = AlertConfig(
            confidence_threshold=0.70,
            alerts_directory=alerts_dir,
            csv_filename="test_metadata.csv",
        )
        
        api_client = ApiClient(ApiConfig(base_url="http://localhost:8000"))
        alert_service = AlertService(alert_cfg, api_client)
        
        now = datetime.now()
        event = DetectionEvent(
            event_type=EventType.ANOMALY,
            confidence=0.85,
            timestamp=now,
            track_id=123,
            class_name="Fighting",
            bbox=(10, 20, 100, 200),
        )
        
        dummy_evidence = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Dispatch event (will fail API upload but should queue and save locally)
        alert_service.evaluate(event, dummy_evidence)
        
        # Check that visual evidence was saved
        saved_images = [f for f in os.listdir(alerts_dir) if f.endswith(".jpg")]
        self.assertEqual(len(saved_images), 1)
        self.assertTrue(saved_images[0].startswith("alert_track_123_"))
        
        # Check CSV file creation and metadata entry
        csv_path = os.path.join(alerts_dir, "test_metadata.csv")
        self.assertTrue(os.path.exists(csv_path))
        
        with open(csv_path, mode="r") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 2)  # Header + 1 row
            self.assertEqual(reader[1][1], "123")  # track_id
            self.assertEqual(reader[1][2], "Fighting")  # class_name
            self.assertEqual(reader[1][3], "0.8500")  # confidence
            self.assertIn(reader[1][7], ["Uploaded", "Pending"])  # upload_status
            
        shutil.rmtree(alerts_dir)


if __name__ == "__main__":
    unittest.main()
