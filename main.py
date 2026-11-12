
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs" / "v2" / "weights" / "best.pt"

CONFIDENCE = 0.5


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    print("Model classes:", model.names)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Camera could not be opened.")

    print("Camera started. Press 'q' to exit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to read camera frame.")
                break

            results = model.predict(
                source=frame,
                conf=CONFIDENCE,
                verbose=False
            )

            annotated_frame = results[0].plot()

            cv2.imshow("Hazard Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
