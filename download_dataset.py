
import os

from roboflow import Roboflow


WORKSPACE = "akram-akbarzade"
PROJECT = "construction-site-safety-4yr9y"
VERSION = 1


def download_dataset():
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not configured.")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    dataset = project.version(VERSION).download("yolov8")

    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


if __name__ == "__main__":
    download_dataset()
