import os.path
import numpy as np
from ultralytics import YOLO
import folder_paths
from huggingface_hub import hf_hub_download


class Detector:
    def __init__(self):
        model_path = os.path.join(folder_paths.models_dir, "echo_mimic/yolov8m.pt")
        if not os.path.exists(model_path):
            print(f"No yolo pt in echo_mimic,auto download from 'Ultralytics/YOLOv8'")
            hf_hub_download(repo_id="Ultralytics/YOLOv8", filename="yolov8m.pt", local_dir=os.path.join(folder_paths.models_dir, "echo_mimic"))
        self.model = YOLO(model_path)
        self.person_id = 0
        self.conf_thres = 0.25

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.detect(img)

    def detect(self, img: np.ndarray) -> np.ndarray:
        self.model.to("cpu")
        results = self.model(img, conf=self.conf_thres)
        
        detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)
        if detections is not []:
            
            # Filter out only person
            person_detections = detections[detections[:, -1] == self.person_id]
            boxes = person_detections[:, :-2].astype(int)
        else:
            boxes=[]

        return boxes


