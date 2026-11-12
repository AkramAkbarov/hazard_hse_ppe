
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "Construction-Site-Safety-1" / "data.yaml"

EPOCHS = 20
IMAGE_SIZE = 416
BATCH_SIZE = 8


def train_model():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_YAML}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=10,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        workers=4,
        cache=True,
        project=str(ROOT / "runs"),
        name="v2",
        seed=42,
        hsv_v=0.4,
        degrees=5,
        scale=0.5,
        fliplr=0.5,
        close_mosaic=5,
    )


def main():
    print("Starting YOLO training...")
    train_model()
    print("Training completed.")


if __name__ == "__main__":
    main()
