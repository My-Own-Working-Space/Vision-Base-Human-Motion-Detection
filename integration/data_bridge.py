import requests
import json
import time
import random

API_URL = "http://localhost:5004/api/detections"
CAMERA_IDS = ["CAM-101", "CAM-102", "CAM-103", "CAM-104"]
BEHAVIORS = ["Normal", "Loitering", "Running", "Intruding", "Falling"]

def send_detection_to_dashboard(camera_id, behavior, confidence, lat, lon):
    payload = {
        "cameraId": camera_id,
        "behaviorType": behavior,
        "confidenceScore": confidence,
        "imagePath": "/uploads/mock_capture.jpg",
        "latitude": lat,
        "longitude": lon
    }
    headers = {'Content-Type': 'application/json'}
    try:
        print(f"Sending: {camera_id} - {behavior} ({confidence*100:.1f}%)")
        response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            print("Successfully sent to dashboard")
            return True
        else:
            print(f"Failed to send: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to dashboard: {e}")
        return False

def simulate_monitoring():
    print("Starting simulated security patrol monitoring...")
    base_lat = 21.0285
    base_lon = 105.8542
    for i in range(10):
        curr_lat = base_lat + random.uniform(-0.01, 0.01)
        curr_lon = base_lon + random.uniform(-0.01, 0.01)
        camera = random.choice(CAMERA_IDS)
        behavior = random.choice(BEHAVIORS)
        conf = random.uniform(0.75, 0.99)
        send_detection_to_dashboard(camera, behavior, conf, curr_lat, curr_lon)
        time.sleep(3)

if __name__ == "__main__":
    simulate_monitoring()
