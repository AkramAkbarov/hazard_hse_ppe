
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

DATA_YAML = ROOT / "Construction-Site-Safety-1" / "data.yaml"

IMAGE_SIZE = 416


def find_best_model():
    model_files = list(ROOT.glob("runs/**/weights/best.pt"))

    if not model_files:
        raise FileNotFoundError("No trained model was found.")

    return max(
        model_files,
        key=lambda path: path.stat().st_mtime
    )


def evaluate_model():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_YAML}")

    with DATA_YAML.open("r", encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file)

    split = "test" if dataset_config.get("test") else "val"

    print(f"Evaluation split: {split}")

    model_path = find_best_model()
    print(f"Using model: {model_path}")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=IMAGE_SIZE,
        device="cpu",
        split=split
    )

    print("\nEvaluation results")
    print("-" * 45)

    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")

    print("\nClass performance")
    print("-" * 45)

    for class_id, score in zip(
        metrics.box.ap_class_index,
        metrics.box.maps[metrics.box.ap_class_index]
    ):
        class_name = model.names[int(class_id)]
        print(f"{class_name:<22} mAP50-95: {score:.3f}")


def main():
    evaluate_model()


if __name__ == "__main__":
    main()
