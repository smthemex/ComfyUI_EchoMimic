# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import time
from enum import Enum
from typing import List
import cv2
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from torchvision import transforms
from .detector import Detector
from .pose_classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_WHOLEBODY_KPTS_COLORS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    GOLIATH_KEYPOINTS,
    GOLIATH_CLASSES_FIX
)



def create_preprocessor(input_size: tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               transforms.Lambda(lambda x: x.unsqueeze(0))
                               ])


def pose_estimation_preprocessor(input_size: tuple[int, int],
                                 mean: List[float] = (0.485, 0.456, 0.406),
                                 std: List[float] = (0.229, 0.224, 0.225)):
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               ])



class TaskType(Enum):
    DEPTH = "depth"
    NORMAL = "normal"
    SEG = "seg"
    POSE = "pose"

class SapiensPoseEstimationType(Enum):
    POSE_ESTIMATION_03B = "sapiens_0.3b_goliath_best_goliath_AP_573.pth"
    POSE_ESTIMATION_06B = "ssapiens_0.6b_goliath_best_goliath_AP_609.pth"
    POSE_ESTIMATION_1B = "sapiens_1b_goliath_best_goliath_AP_639.pth"
    POSE_ESTIMATION_03B_T = "sapiens_0.3b_goliath_best_goliath_AP_575_torchscript.pt2"
    POSE_ESTIMATION_06B_T = "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2"
    POSE_ESTIMATION_1B_T = "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"
    POSE_ESTIMATION_03B_16 = "sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2"
    POSE_ESTIMATION_06B_16 = "sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2"
    POSE_ESTIMATION_1B_16 = "sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2"
    OFF="off"
    

class SapiensPoseEstimation:
    def __init__(self,
                 type: SapiensPoseEstimationType = SapiensPoseEstimationType.POSE_ESTIMATION_1B_T, local_pose="",pt_type="float32_torch",model_dir="",img_size=(1024, 768),use_torchscript=True,show_pose_object=False,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):
        # Load the model
        self.local_pose = local_pose
        self.model_dir = model_dir
        self.pt_type = pt_type
        self.img_size = img_size
        self.use_torchscript = use_torchscript
        self.show_pose_object=show_pose_object
        if self.local_pose:
            path = self.local_pose
        else:
            path = download_hf_model(type.value, TaskType.POSE, self.model_dir, dtype=self.pt_type)
        self.device = device
        self.dtype = dtype
        if self.use_torchscript:
            model = torch.jit.load(path)
            model = model.eval()
            model.to(device).to(dtype)
        else:
            model = torch.export.load(path).module()
            dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.half
            model.to(dtype)
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
        self.model = model
        self.preprocessor = pose_estimation_preprocessor(input_size=self.img_size)

        # Initialize the YOLO-based detector
        self.detector = Detector()

    def __call__(self, img: np.ndarray,filter_obj):
        #start = time.perf_counter()

        # Detect persons in the image
        bboxes = self.detector.detect(img)
        
        GOLIATH_HAND_KEY = ["hand", "finger", "thumb", "wrist", ]
        GOLIATH_FACE_NECK_KEY = ["nose", "eye", "neck", "ear", "labiomental", "glabella", "chin", "lash", "crease",
                                 "nostril", "mouth", "lip", "helix", "tragus", "iris", "pupil", "between_22_15",
                                 "concha", "crus"]
        GOLIATH_LOWER_LIMBS_KEY = ["hip", "knee", "ankle", "toe", "heel"]
        GOLIATH_TORSO_KEY = ["hip", "shoulder"]
        GOLIATH_ELBOW_HAND_KEY = ["hand", "finger", "thumb", "elbow", "wrist", "olecranon", "cubital_fossa"]
        filter_obj_list = {"Face_Neck": GOLIATH_FACE_NECK_KEY, "Left_Hand": GOLIATH_HAND_KEY,
                           "Left_Foot": GOLIATH_LOWER_LIMBS_KEY, "Torso": GOLIATH_TORSO_KEY,
                           "Left_Lower_Arm": GOLIATH_ELBOW_HAND_KEY}
        filter_obj_done = []
        if filter_obj:
            select_str_list = [list(GOLIATH_CLASSES_FIX)[i].split(".")[-1] for i in filter_obj]
            if all(any(x in y for y in select_str_list) for x in list(filter_obj_list.keys())):
                filter_obj_done = []
            else:
                for i in select_str_list:
                    if filter_obj_list.get(i):
                        filter_obj_done.append(filter_obj_list.get(i))
            
            # Process the image and estimate the pose
        pose_result_image, keypoints, box_size = self.estimate_pose(img, bboxes, filter_obj_done)
       

        #print(f"Pose estimation inference took: {time.perf_counter() - start:.4f} seconds")
        return pose_result_image, keypoints,box_size
    
    def enable_model_cpu_offload(self):
        self.model.to("cpu")
        torch.cuda.empty_cache()
        
    def move_to_cuda(self):
        self.model.to("cuda")

    @torch.inference_mode()
    def estimate_pose(self, img: np.ndarray, bboxes: List[List[float]],filter_obj_done) -> (np.ndarray, List[dict],tuple):
        all_keypoints = []
        result_img = img.copy()
        box_size=[]
        for bbox in bboxes:
            cropped_img = self.crop_image(img, bbox)
            tensor = self.preprocessor(cropped_img).unsqueeze(0).to(self.device).to(self.dtype)

            heatmaps = self.model(tensor).to(torch.float32)
            keypoints = self.heatmaps_to_keypoints(heatmaps[0].cpu().numpy())
            all_keypoints.append(keypoints)
            if not self.show_pose_object:
                # Draw black BG
                empty_cv = np.empty(result_img.shape, dtype=np.uint8)
                empty_cv[:] = (0, 0, 0)
                result_img,box_size = self.draw_keypoints(empty_cv, keypoints, bbox,filter_obj_done)
            else:
                # Draw the keypoints on the original image
                result_img,box_size = self.draw_keypoints(result_img, keypoints, bbox,filter_obj_done)
           
        return result_img, all_keypoints,box_size

    def crop_image(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox[:4])
        return img[y1:y2, x1:x2]


    def heatmaps_to_keypoints(self, heatmaps: np.ndarray) -> dict:
        keypoints = {}
        for i, name in enumerate(GOLIATH_KEYPOINTS):
            if i < heatmaps.shape[0]:
                y, x = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
                conf = heatmaps[i, y, x]
                keypoints[name] = (float(x), float(y), float(conf))
        return keypoints

 
    def draw_keypoints(self, img: np.ndarray, keypoints: dict, bbox: List[float],filter_obj_done) ->(np.ndarray,tuple) :
        
        if filter_obj_done:
            get_list=[]
            for i in filter_obj_done:
                for j in i:
                    get_list.append(j)
            obj_list=list(set(get_list))
            KEYPOINTS_obj = {}
            for i in obj_list:
                for j, (name, (x, y, conf)) in enumerate(keypoints.items()):
                    if i in name:
                        KEYPOINTS_obj[name]=(x, y, conf)
            keypoints=KEYPOINTS_obj
            
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        box_xy=x1, y1, x2, y2
        
        bbox_width, bbox_height = x2 - x1, y2 - y1
       
        img_copy = img.copy()
        if bbox is []:
            box_xy=None
        #box_size=(bbox_width, bbox_height)
        # Draw keypoints on t1Bhe image
        for i, (name, (x, y, conf)) in enumerate(keypoints.items()):
            if conf > 0.3:  # Only draw confident keypoints
                x_coord = int(x * bbox_width / 192) + x1
                y_coord = int(y * bbox_height / 256) + y1
                cv2.circle(img_copy, (x_coord, y_coord), 3, GOLIATH_KPTS_COLORS[i], -1)
        
        upper_limb_keypoints=[]
        # Optionally draw skeleton
        for _, link_info in GOLIATH_SKELETON_INFO.items():
            pt1_name, pt2_name = link_info['link']
            if pt1_name in keypoints and pt2_name in keypoints:
                pt1 = keypoints[pt1_name]
                pt2 = keypoints[pt2_name]
                if pt1[2] > 0.3 and pt2[2] > 0.3:
                    x1_coord = int(pt1[0] * bbox_width / 192) + x1
                    y1_coord = int(pt1[1] * bbox_height / 256) + y1
                    x2_coord = int(pt2[0] * bbox_width / 192) + x1
                    y2_coord = int(pt2[1] * bbox_height / 256) + y1
                    cv2.line(img_copy, (x1_coord, y1_coord), (x2_coord, y2_coord), GOLIATH_KPTS_COLORS[i], 2)

        return img_copy,box_xy


def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            
            # tqdm has many interesting parameters. Feel free to experiment!
            tqdm_params = {
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)


def download_hf_model(model_name: str, task_type: TaskType, model_dir: str = 'models', dtype="float32"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    path = model_dir + f"/{task_type.value}/" + model_name
    if os.path.exists(path):
        return path
    
    print(f"Model {model_name} not found, downloading from Hugging Face Hub...")
    
    if "0.3b" in model_name:
        model_version = "0.3b"
    elif "0.6b" in model_name:
        model_version = "0.6b"
    elif "1b" in model_name:
        model_version = "1b"
    elif "2b" in model_name:
        model_version = "2b"
    else:
        raise "get unsupport model_name"
    
    if dtype == "float32":
        repo_id = f"facebook/sapiens-{task_type.value}-{model_version}"
    elif dtype == "float32_torch":
        repo_id = f"facebook/sapiens-{task_type.value}-{model_version}-torchscript"  # torchscript
    else:
        repo_id = f"facebook/sapiens-{task_type.value}-{model_version}-{dtype}"  # bfloat16
    
    real_dir = os.path.join(model_dir, f"{task_type.value}")
    
    hf_hub_download(repo_id=repo_id, filename=model_name, local_dir=real_dir)
    print("Model downloaded successfully to", path)
    return path
