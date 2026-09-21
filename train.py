from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="Construction-Site-Safety-1/data.yaml",
    epochs=30,
    imgsz=416,
    batch=8,
    device="cpu",
    project="runs",
    name="hazard_v1"
)