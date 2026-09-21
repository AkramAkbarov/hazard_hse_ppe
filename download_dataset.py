from roboflow import Roboflow

rf = Roboflow(api_key="Roboflow_api_key")
project = rf.workspace("akram-akbarzade").project("construction-site-safety-4yr9y")
version = project.version(1)
dataset = version.download("yolov8")
