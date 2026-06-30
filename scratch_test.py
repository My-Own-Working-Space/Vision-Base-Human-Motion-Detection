import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from edge.clients.pms_bridge_client import PmsBridgeClient
from edge.models import AlertPayload

from dotenv import load_dotenv
load_dotenv()

pms_url = os.getenv("PMS_BRIDGE_URL", "http://188.166.191.199:5194")
client = PmsBridgeClient(pms_base_url=pms_url)

payload = AlertPayload(
    class_name="Corrosion",
    confidence=0.9654,
    timestamp=datetime.utcnow().isoformat(),
    latitude=10.7769,
    longitude=106.7009,
    image_path="alerts/alert_track_2_20260526_172904_540897.jpg",
    image_name="alert_track_2_20260526_172904_540897.jpg",
    track_id=2
)

success = client.send_detection(payload, drone_id="UAV001")
print(f"FORWARD SUCCESS: {success}")
