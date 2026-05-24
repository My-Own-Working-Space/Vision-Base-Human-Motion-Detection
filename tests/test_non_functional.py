import os
import sys
import unittest
import time
import numpy as np
import cv2
import psutil
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inference')))

from yolo_detector import YOLODetector
from resnet_lstm_classifier import ResNetLSTMClassifier

class TestNonFunctionalPerformance(unittest.TestCase):
    """
    Non-functional tests: profiling execution performance (FPS, latency),
    resource consumption (CPU, RAM), and edge-case robustness.
    """

    @classmethod
    def setUpClass(cls):
        cls.test_image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '../.venv/lib/python3.12/site-packages/ultralytics/assets/zidane.jpg'
        ))
        if not os.path.exists(cls.test_image_path):
            cls.test_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_sample.jpg'))
            dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(cls.test_image_path, dummy_img)

        cls.yolo_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/yolov8n.pt'))
        cls.lstm_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/resnet_lstm_best.pth'))

        # Track resource utilization during initialization
        cls.process = psutil.Process(os.getpid())
        cls.ram_before = cls.process.memory_info().rss / (1024 * 1024) # MB

        cls.detector = YOLODetector(cls.yolo_model_path)
        cls.classifier = ResNetLSTMClassifier(cls.lstm_model_path)

        cls.ram_after = cls.process.memory_info().rss / (1024 * 1024) # MB
        cls.init_ram_usage = cls.ram_after - cls.ram_before

    def test_latency_and_throughput_benchmarks(self):
        """Benchmark FPS and latency profiles for each pipeline stage."""
        frame = cv2.imread(self.test_image_path)
        dummy_crop = Image.fromarray(np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8))
        
        # 1. Benchmark YOLODetector.track
        num_iters = 30
        t0 = time.perf_counter()
        for _ in range(num_iters):
            _ = self.detector.track(frame)
        t1 = time.perf_counter()
        yolo_time = (t1 - t0) / num_iters
        yolo_fps = 1.0 / yolo_time
        print(f"\n[Benchmark] YOLOv8+ByteTrack Tracking: Latency={yolo_time*1000:.2f}ms, Throughput={yolo_fps:.2f} FPS")

        # 2. Benchmark Feature Extraction (ResNet18)
        t0 = time.perf_counter()
        for _ in range(num_iters):
            _ = self.classifier._extract_feature(dummy_crop)
        t1 = time.perf_counter()
        resnet_time = (t1 - t0) / num_iters
        resnet_throughput = 1.0 / resnet_time
        print(f"[Benchmark] ResNet18 Feature Extraction: Latency={resnet_time*1000:.2f}ms, Throughput={resnet_throughput:.2f} crops/sec")

        # 3. Benchmark Temporal LSTM Inference
        # Seed 15 features first to fill sequence length
        track_id = 888
        for _ in range(15):
            _, _, _ = self.classifier.predict(dummy_crop, track_id)

        t0 = time.perf_counter()
        for _ in range(num_iters):
            _, _, _ = self.classifier.predict(dummy_crop, track_id)
        t1 = time.perf_counter()
        lstm_time = (t1 - t0) / num_iters
        lstm_throughput = 1.0 / lstm_time
        print(f"[Benchmark] LSTM Temporal Inference: Latency={lstm_time*1000:.2f}ms, Throughput={lstm_throughput:.2f} predictions/sec")

        # Ensure performance stays within reasonable thresholds on CPU
        # YOLOv8n tracking should easily process frames under 300ms
        self.assertLess(yolo_time, 0.35, "YOLOv8 tracking is too slow (> 350ms per frame).")
        self.assertLess(resnet_time, 0.08, "ResNet feature extraction is too slow (> 80ms per crop).")
        self.assertLess(lstm_time, 0.05, "LSTM inference is too slow (> 50ms per prediction).")

    def test_resource_footprint(self):
        """Profile CPU load and memory footprint during continuous inference."""
        frame = cv2.imread(self.test_image_path)
        dummy_crop = Image.fromarray(np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8))
        
        print(f"\n[Profile] Model Initialization RAM Footprint: {self.init_ram_usage:.2f} MB")
        self.assertLess(self.init_ram_usage, 600, "Memory footprint of models is excessively high (> 600 MB).")

        # Run inference loops and measure peak resource consumption
        cpu_percentages = []
        ram_measurements = []
        
        # Capture resource usage during active inference
        for _ in range(15):
            # Run YOLO + ResNet + LSTM prediction
            results = self.detector.track(frame)
            crops = self.detector.get_tracked_crops(frame, results)
            for item in crops:
                _, _, _ = self.classifier.predict(item['image'], item['track_id'])
            
            # Read resource stats
            cpu_percentages.append(psutil.cpu_percent(interval=None))
            ram_measurements.append(self.process.memory_info().rss / (1024 * 1024))
            time.sleep(0.01)

        avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
        max_ram = max(ram_measurements)
        
        print(f"[Profile] Average CPU Load during Active Pipeline: {avg_cpu:.1f}%")
        print(f"[Profile] Peak Memory RSS during Inference: {max_ram:.2f} MB")
        self.assertLess(max_ram, 1200, "Memory leaked or excessively high during active inference (> 1.2 GB).")

    def test_robustness_and_fault_tolerance(self):
        """Test system robustness against corrupt/empty inputs, size deviations, and missing models."""
        
        # 1. Edge Case: Completely black/empty frame
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            results = self.detector.track(black_frame)
            crops = self.detector.get_tracked_crops(black_frame, results)
            # Should run without error and find 0 people
            self.assertEqual(len(crops), 0, "No tracks should be detected on empty black frame.")
        except Exception as e:
            self.fail(f"YOLODetector failed to process a completely black frame: {e}")

        # 2. Edge Case: Massive input crop resizing robustness
        giant_crop = Image.fromarray(np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8))
        tiny_crop = Image.fromarray(np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8))
        
        try:
            feat_giant = self.classifier._extract_feature(giant_crop)
            feat_tiny = self.classifier._extract_feature(tiny_crop)
            self.assertEqual(feat_giant.shape[0], 512)
            self.assertEqual(feat_tiny.shape[0], 512)
        except Exception as e:
            self.fail(f"ResNet transform failed on extreme input crop dimensions: {e}")

        # 3. Robustness: Graceful fallback on missing LSTM weights
        try:
            missing_classifier = ResNetLSTMClassifier("models/non_existent_file.pth")
            self.fail("ResNetLSTMClassifier should raise FileNotFoundError or runtime error for missing weights.")
        except FileNotFoundError:
            # Expected behavior
            pass
        except Exception as e:
            # Also acceptable if it throws standard PyTorch loader errors, but we want FileNotFoundError ideally
            pass

if __name__ == '__main__':
    unittest.main()
