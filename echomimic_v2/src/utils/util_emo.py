import torch
import cv2
import sys
import numpy as np
import os
from PIL import Image
# from zdete import Predictor as BboxPredictor
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class MyWav2Vec():
    def __init__(self, model_path, device="cuda"):
        super(MyWav2Vec, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.wav2Vec = Wav2Vec2Model.from_pretrained(model_path).to(device)
        self.device = device
        print("### Wav2Vec model loaded ###")

    def forward(self, x):
        return self.wav2Vec(x).last_hidden_state

    def process(self, x):
        return self.processor(x, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)

class AutoFlow():
    def __init__(self, auto_flow_dir, imh=512, imw=512):
        super(AutoFlow, self).__init__()
        model_dir = auto_flow_dir+'/third_lib/model_zoo/'
        cfg_file = model_dir + '/zdete_detector/mobilenet_v1_0.25.yaml'
        model_file = model_dir + '/zdete_detector/last_39.pt'
        self.bbox_predictor = BboxPredictor(cfg_file, model_file, imgsz=320, conf_thres=0.6, iou_thres=0.2)
        self.imh = imh
        self.imw = imw
        print("### AutoFlow bbox_predictor loaded ###")

    def frames_to_face_regions(self, frames, toPIL=True):
        # 输入是bgr numpy格式
        face_region_list = []
        for img in frames:
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bbox = self.bbox_predictor.predict(img)[0][0]
            xyxy = bbox[:4]
            score = bbox[4]
            xyxy = np.round(xyxy).astype('int')
            rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
            face_mask = np.zeros((img.shape[0], img.shape[1])).astype('uint8')
            face_mask[rb:re,cb:ce] = 255
            face_mask = cv2.resize(face_mask, ((self.imw, self.imh)))
            if toPIL:
                face_mask = Image.fromarray(face_mask)
            face_region_list.append(face_mask)
        return face_region_list

def xyxy2x0y0wh(bbox):
    x0, y0, x1, y1 = bbox[:4]
    return [x0, y0, x1-x0, y1-y0]

def video_to_frame(video_path: str, interval=1, max_frame=None, imh=None, imw=None, is_return_sum=False, is_rgb=False):
    vidcap = cv2.VideoCapture(video_path) 
    success = True

    key_frames = []
    sum_frames = None
    count = 0
    while success:
        success, image = vidcap.read()
        if image is not None:
            if is_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if imh is not None and imw is not None:
                image = img_resize(image, imh=None, imw=None)

            if count % interval == 0:
                key_frames.append(image)
                if is_return_sum:
                    if sum_frames is None:
                        sum_frames = image.copy().astype('float32')
                    else:
                        sum_frames = sum_frames + image

            count += 1

            if max_frame != None:
                if count >= max_frame:
                    break

    vidcap.release()
    if is_return_sum:
        return key_frames, sum_frames
    else:
        return key_frames

def img_resize(input_img, imh=None, imw=None, max_val=512):
    if imh is not None and imw is not None:
        width, height = imw, imh
    else:
        height, width = input_img.shape[0], input_img.shape[1]
        if height > width:
            ratio = width/height
            height = max_val
            width = ratio * height
        else:
            ratio = height/width
            width = max_val
            height = ratio * width

        height = int(round(height/8)*8)
        width = int(round(width/8)*8)

    input_img = cv2.resize(input_img, (width, height))
    return input_img

def assign_audio_to_frame(audio_input, frame_num):
    audio_len = audio_input.shape[0]
    audio_per_frame = audio_len / frame_num
    audio_to_frame_list = []
    for f_i in range(frame_num):
        start_idx = int(round(f_i * audio_per_frame))
        end_idx = int(round((f_i + 1) * audio_per_frame))
        if start_idx >= audio_len:
            start_idx = int(round(start_idx - audio_per_frame))
        # print(f"frame_i:{f_i}, start_index:{start_idx}, end_index:{end_idx}, audio_length:{audio_input.shape}")
        seg_audio = audio_input[start_idx:end_idx, :]
        if type(seg_audio) == np.ndarray:
            seg_audio = seg_audio.mean(axis=0, keepdims=True) # B * 20 * 768
        elif torch.is_tensor(seg_audio):
            seg_audio = seg_audio.mean(dim=0, keepdim=True)
        audio_to_frame_list.append(seg_audio)

    if type(seg_audio) == np.ndarray:
        audio_to_frames = np.concatenate(audio_to_frame_list, 0)
    else:
        audio_to_frames = torch.cat(audio_to_frame_list, 0)
    return audio_to_frames

def assign_audio_to_frame_new(audio_input, frame_num, pad_frame):
    audio_len = audio_input.shape[0]
    audio_to_frame_list = []
    for f_i in range(frame_num):
        mid_index = int(f_i / frame_num * audio_len)
        start_idx = mid_index - pad_frame
        end_idx = mid_index + pad_frame + 1

        if start_idx < 0:
            start_idx = 0
            end_idx = start_idx + pad_frame * 2 + 1
        if end_idx >= audio_len:
            end_idx = audio_len - 1
            start_idx = end_idx - (pad_frame * 2 + 1)

        seg_audio = audio_input[None, start_idx:end_idx, :]
        audio_to_frame_list.append(seg_audio)

    if type(seg_audio) == np.ndarray:
        audio_to_frames = np.concatenate(audio_to_frame_list, 0)
    else:
        audio_to_frames = torch.cat(audio_to_frame_list, 0)
    return audio_to_frames

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value