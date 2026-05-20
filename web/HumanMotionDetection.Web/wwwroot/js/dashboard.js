document.addEventListener('DOMContentLoaded', function () {
    var map = L.map('map').setView([21.0285, 105.8542], 13);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    var markers = [];

    async function loadDetections() {
        try {
            const response = await fetch('/api/detections');
            const data = await response.json();
            updateDashboard(data);
        } catch (error) {
            console.error('Error fetching detections:', error);
        }
    }

    function updateDashboard(data) {
        markers.forEach(m => map.removeLayer(m));
        markers = [];
        
        const rows = document.getElementById('detection-rows');
        const countBadge = document.getElementById('threat-count');
        
        rows.innerHTML = '';
        countBadge.innerText = data.filter(d => d.behaviorType !== 'Normal').length;

        data.forEach(det => {
            if (det.latitude && det.longitude) {
                var color = det.behaviorType === 'Normal' ? '#10b981' : '#ef4444';
                
                var markerIcon = L.circleMarker([det.latitude, det.longitude], {
                    radius: 8,
                    fillColor: color,
                    color: "#fff",
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.8
                }).addTo(map);
                
                markerIcon.bindPopup(`<b>Camera: ${det.cameraId}</b><br>Activity: ${det.behaviorType}<br>Conf: ${Math.round(det.confidenceScore * 100)}%`);
                markers.push(markerIcon);
            }

            const row = `
                <tr>
                    <td>${det.id}</td>
                    <td><strong>${det.cameraId}</strong></td>
                    <td>
                        <span class="badge ${getBadgeClass(det.behaviorType)}">${det.behaviorType}</span>
                    </td>
                    <td>${Math.round(det.confidenceScore * 100)}%</td>
                    <td>${new Date(det.createdDate).toLocaleString()}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-info" onclick="viewDetail(${det.id})">Analyze</button>
                    </td>
                </tr>
            `;
            rows.innerHTML += row;
        });
    }

    function getBadgeClass(type) {
        if (type === 'Normal') return 'bg-success';
        if (type === 'Intruding') return 'bg-danger animate-pulse';
        if (type === 'Loitering') return 'bg-warning text-dark';
        return 'bg-info text-dark';
    }

    window.viewDetail = function(id) {
        alert("Analyzing activity logs for ID: " + id);
    };

    loadDetections();
    setInterval(loadDetections, 10000);
});
