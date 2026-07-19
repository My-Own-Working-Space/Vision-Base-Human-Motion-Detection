from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from edge.clients.api_client import ApiClient
from edge.config import load_config
from edge.main import EdgeRuntimeServices, create_app
from edge.models import DetectionEvent, EventType
from edge.services.alert_service import AlertService
from edge.ui.stream_renderer import StreamRenderer


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_IMAGE = ROOT / "alerts" / "alert_track_2_20260526_172904_540897.jpg"


class FakeCameraService:
    frame_index = 0

    def open(self) -> bool:
        return True

    def close(self) -> None:
        pass


class FakeDetectionService:
    has_classifier = True

    def __init__(self) -> None:
        self.initialize_calls = 0
        self.process_calls = 0
        self.force_predict_values: list[bool] = []

    def initialize(self) -> None:
        self.initialize_calls += 1

    def process_frame(self, frame, force_predict: bool = False):
        self.process_calls += 1
        self.force_predict_values.append(force_predict)
        h, w = frame.shape[:2]
        event = DetectionEvent(
            event_type=EventType.ANOMALY if force_predict else EventType.NORMAL,
            confidence=0.91 if force_predict else 0.73,
            timestamp=datetime(2026, 7, 19, 12, 0, self.process_calls % 60),
            track_id=100 + self.process_calls,
            class_name="FakeAnomaly" if force_predict else "FakeNormal",
            buffer_length=16,
            bbox=(1, 2, min(w, 20), min(h, 30)),
        )
        return [(event, None)]


def make_services(fake_detection: FakeDetectionService) -> EdgeRuntimeServices:
    config = load_config()
    api_client = ApiClient(config.api)
    return EdgeRuntimeServices(
        config=config,
        api_client=api_client,
        camera_service=FakeCameraService(),
        detection_service=fake_detection,
        alert_service=AlertService(config.alert, api_client),
        renderer=StreamRenderer(sequence_length=config.detection.sequence_length),
    )


def make_client(fake_detection: FakeDetectionService) -> TestClient:
    app = create_app(
        services=make_services(fake_detection),
        initialize_models=False,
        start_background_workers=False,
    )
    return TestClient(app)


def make_test_video_bytes() -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.avi"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            4.0,
            (64, 48),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create test video")
        for idx in range(6):
            frame = np.full((48, 64, 3), idx * 20, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path.read_bytes()


class AnalyzeApiFlowTests(unittest.TestCase):
    def test_imported_app_does_not_start_heavy_side_effects(self) -> None:
        import edge.main as main_module

        services = main_module.app.state.services
        self.assertFalse(services.models_initialized)
        self.assertFalse(services.alert_worker_started)
        self.assertIsNone(services.pms_thread)

    def test_single_image_upload_detects_and_returns_result(self) -> None:
        fake_detection = FakeDetectionService()
        with make_client(fake_detection) as client:
            response = client.post(
                "/api/analyze",
                data={"analysis_type": "HumanMotionDetection"},
                files={"file": ("fixture.jpg", FIXTURE_IMAGE.read_bytes(), "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mediaType"], "Image")
        self.assertEqual(body["summary"]["anomalyCount"], 1)
        self.assertEqual(body["detections"][0]["className"], "FakeAnomaly")
        self.assertEqual(fake_detection.force_predict_values, [True])
        self.assertEqual(fake_detection.initialize_calls, 0)

    def test_single_video_upload_detects_and_returns_result(self) -> None:
        fake_detection = FakeDetectionService()
        video_bytes = make_test_video_bytes()
        with make_client(fake_detection) as client:
            response = client.post(
                "/api/analyze",
                data={"analysis_type": "HumanMotionDetection"},
                files={"file": ("sample.avi", video_bytes, "video/x-msvideo")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mediaType"], "Video")
        self.assertGreaterEqual(len(body["detections"]), 1)
        self.assertIn("frameIndex", body["detections"][0])
        self.assertTrue(all(value is False for value in fake_detection.force_predict_values))

    def test_batch_upload_accepts_many_images_and_videos(self) -> None:
        fake_detection = FakeDetectionService()
        video_bytes = make_test_video_bytes()
        files = [
            ("files", ("one.jpg", FIXTURE_IMAGE.read_bytes(), "image/jpeg")),
            ("files", ("two.jpg", FIXTURE_IMAGE.read_bytes(), "image/jpeg")),
            ("files", ("clip.avi", video_bytes, "video/x-msvideo")),
        ]
        with make_client(fake_detection) as client:
            response = client.post(
                "/api/analyze-batch",
                data={"analysis_type": "HumanMotionDetection"},
                files=files,
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 3)
        self.assertTrue(all(item["ok"] for item in body["items"]))
        self.assertEqual(body["items"][0]["result"]["mediaType"], "Image")
        self.assertEqual(body["items"][2]["result"]["mediaType"], "Video")
        self.assertGreaterEqual(body["summary"]["personsDetected"], 3)

    def test_unsupported_upload_returns_400(self) -> None:
        fake_detection = FakeDetectionService()
        with make_client(fake_detection) as client:
            response = client.post(
                "/api/analyze",
                files={"file": ("bad.txt", b"not media", "text/plain")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["error"])
        self.assertEqual(fake_detection.process_calls, 0)


if __name__ == "__main__":
    unittest.main()
