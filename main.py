import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/runs/hazard_v1-2/weights/best.pt")

print("Model class-ları:", model.names)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılmadı!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.5)

    frame = results[0].plot()

    cv2.imshow("Hazard Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()