# Vision-Based Human Motion and Suspicious Behavior Detection

He thong nhan dien con nguoi va hanh vi kha nghi su dung AI ket hop YOLOv8 va Custom CNN MobileNetV3.

## Tinh nang chinh

- YOLOv8 Detector: Xac dinh vi tri con nguoi trong khung hinh.
- Custom CNN Refiner: Su dung MobileNetV3 de phan loai chi tiet hanh vi nhu di bo binh thuong, lo lo dang ngo, chay tron, dot nhap, nga quy.
- IP Camera Integration: Ket noi voi thiet bi smartphone hoac camera IP thoi gian thuc.
- FastAPI Backend: Xuly va truyen tai video thoi gian thuc da duoc phan tich.
- ASP.NET Core Dashboard: Giao dien web hien dai giam sat an ninh truc quan voi ban do va he thong canh bao tuc thi.

## Cau truc thu muc

- /training: Ma nguon huan luyen mo hinh YOLOv8 va CNN.
- /inference: Giao dien nhan dien thuc te phat stream tu FastAPI.
- /models: Luu tru cac trong so mo hinh pth va pt.
- /web: Dashboard quan ly an ninh viet bang ASP.NET Core 9.

## Huong dan chay he thong

### 1. Cai dat thu vien Python

pip install fastapi uvicorn opencv-python ultralytics torch torchvision pillow requests

### 2. Khoi chay Web API va Dashboard

cd web/HumanMotionDetection.Web
dotnet run

Dashboard se hoat dong tai http://localhost:5004

### 3. Khoi chay FastAPI Real-time Stream

cd inference
python app.py

FastAPI API se hoat dong tai http://localhost:8000