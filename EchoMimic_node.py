# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import pickle
import random
import sys
from pathlib import Path
from shutil import copyfile, copy
import cv2
import numpy as np
import torch
import torchaudio
from diffusers import AutoencoderKL,DDIMScheduler
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from .src.models.unet_2d_condition import UNet2DConditionModel
from .src.models.unet_3d_echo import EchoUNet3DConditionModel
from .src.models.whisper.audio2feature import load_audio_model
from .src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from .src.pipelines.pipeline_echo_mimic_pose import AudioPose2VideoPipeline
from .src.pipelines.pipeline_echo_mimic_pose_acc import AudioPose2VideoPipeline as AudioPose2VideoaccPipeline
from .src.utils.util import save_videos_grid, crop_and_pad
from .src.models.face_locator import FaceLocator
from .src.utils.draw_utils import FaceMeshVisualizer
from .src.utils.mp_utils  import LMKExtractor
from .src.utils.img_utils import pil_to_cv2, cv2_to_pil, center_crop_cv2, pils_from_video, save_videos_from_pils, save_video_from_cv2_list
from .src.utils.motion_utils import motion_sync
from moviepy.editor import VideoFileClip, AudioFileClip
from facenet_pytorch import MTCNN
import copy
import pathlib
import pickle
from glob import glob
import folder_paths
from comfy.utils import common_upscale
import platform
import subprocess


MAX_SEED = np.iinfo(np.int32).max
current_path = os.path.dirname(os.path.abspath(__file__))
node_path_dir = os.path.dirname(current_path)
comfy_file_path = os.path.dirname(node_path_dir)
weigths_current_path = os.path.join(folder_paths.models_dir, "echo_mimic")

if not os.path.exists(weigths_current_path):
    os.makedirs(weigths_current_path)
    
weigths_uet_current_path = os.path.join(weigths_current_path, "unet")
if not os.path.exists(weigths_uet_current_path):
    os.makedirs(weigths_uet_current_path)

weigths_au_current_path = os.path.join(weigths_current_path, "audio_processor")
if not os.path.exists(weigths_au_current_path):
    os.makedirs(weigths_au_current_path)

tensorrt_lite= os.path.join(folder_paths.input_directory,"tensorrt_lite")
if not os.path.exists(tensorrt_lite):
    os.makedirs(tensorrt_lite)

def find_directories(base_path):
    directories = []
    for root, dirs, files in os.walk(base_path):
        for name in dirs:
            directories.append(name)
    return directories
   
pose_path_list = find_directories(tensorrt_lite)
if pose_path_list:
    pose_path_list_=["none"]+pose_path_list
else:
    pose_path_list_=["none",]



ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None and platform.system() in ['Linux', 'Darwin']:
    try:
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip()
            print(f"FFmpeg is installed at: {ffmpeg_path}")
        else:
            print("FFmpeg is not installed. Please download ffmpeg-static and export to FFMPEG_PATH.")
            print("For example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
    except Exception as e:
        pass

if ffmpeg_path is not None and ffmpeg_path not in os.getenv('PATH'):
    print("Adding FFMPEG_PATH to PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"
    
# config_path = os.path.join(current_path, "configs/prompts/animation.yaml")
# config = OmegaConf.load(config_path)
# if config.weight_dtype == "fp16":
#     weight_dtype = torch.float16
# else:
#     weight_dtype = torch.float32
weight_dtype = torch.float16

device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"

inference_config_path = os.path.join(current_path,"configs","inference","inference_v2.yaml")
infer_config = OmegaConf.load(inference_config_path)

def select_face(det_bboxes, probs):
    ## max face from faces that the prob is above 0.8
    ## box: xyxy
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None
    sorted_bboxes = sorted(filtered_bboxes, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]

def process_video(uploaded_img, uploaded_audio, width, height, length, seed, facemask_dilation_ratio,
                  facecrop_dilation_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps, device,pipe,face_detector,save_video,pose_dir,visualizer=None,):
    if seed is not None and seed > -1:
        generator = torch.manual_seed(seed)
    else:
        generator = torch.manual_seed(random.randint(100, 1000000))
    
    #### face musk prepare
    face_img=np.array(uploaded_img)
    face_img=cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')
    det_bboxes, probs = face_detector.detect(face_img)
    select_bbox = select_face(det_bboxes, probs)
    if select_bbox is None:
        face_mask[:, :] = 255
    else:
        xyxy = select_bbox[:4].astype(float)# 原方法的np版本 无法实现整形
        xyxy = np.round(xyxy).astype("int")
        rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
        r_pad = int((re - rb) * facemask_dilation_ratio)
        c_pad = int((ce - cb) * facemask_dilation_ratio)
        face_mask[rb - r_pad: re + r_pad, cb - c_pad: ce + c_pad] = 255
        
        #### face crop
        r_pad_crop = int((re - rb) * facecrop_dilation_ratio)
        c_pad_crop = int((ce - cb) * facecrop_dilation_ratio)
        crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]),
                     min(re + r_pad_crop, face_img.shape[0])]
        face_img = crop_and_pad(face_img, crop_rect)
        face_mask = crop_and_pad(face_mask, crop_rect)
        face_img = cv2.resize(face_img, (width, height))
        face_mask = cv2.resize(face_mask, (width, height))
    
    ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
    
    if visualizer:
        pose_list = []
        for index in range(len(os.listdir(pose_dir))):
            tgt_musk_path = os.path.join(pose_dir, f"{index}.pkl")
            with open(tgt_musk_path, "rb") as f:
                tgt_kpts = pickle.load(f)
            tgt_musk = visualizer.draw_landmarks((width, height), tgt_kpts)
            tgt_musk_pil = Image.fromarray(np.array(tgt_musk).astype(np.uint8)).convert('RGB')
            pose_list.append(
                torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device="cuda").permute(2, 0, 1) / 255.0)
        face_mask_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
        ref_image_pil=uploaded_img.convert("RGB")
    else:
        face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device="cuda").unsqueeze(0).unsqueeze(
            0).unsqueeze(0) / 255.0
    
    video = pipe(
        ref_image_pil,
        uploaded_audio,
        face_mask_tensor,
        width,
        height,
        length,
        steps,
        cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=context_frames,
        fps=fps,
        context_overlap=context_overlap
    ).videos
    filename_prefix = ''.join(random.choice("0123456789") for _ in range(5))
    output_file = os.path.join(folder_paths.output_directory, f"{filename_prefix}_echo.mp4")
    ouput_list = save_videos_grid(video, output_file, n_rows=1, fps=fps, save_video=save_video)
    if save_video:
        output_video_path=os.path.join(folder_paths.output_directory,f"{filename_prefix}_audio.mp4")
        video_clip = VideoFileClip(output_file)
        audio_clip = AudioFileClip(uploaded_audio)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(
            output_video_path,
            codec="libx264", audio_codec="aac")
        print(f"saving {output_video_path}")
        video_clip.reader.close()
        audio_clip.close()
        final_clip.reader.close()
    return ouput_list


def download_weights(file_dir,repo_id,subfolder="",pt_name=""):
    if subfolder:
        file_path = os.path.join(file_dir,subfolder, pt_name)
        sub_dir=os.path.join(file_dir,subfolder)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        if not os.path.exists(file_path):
            pt_path = hf_hub_download(
                repo_id=repo_id,
                subfolder=subfolder,
                filename=pt_name,
                local_dir = file_dir,
            )
        else:
            pt_path=get_instance_path(file_path)
        return pt_path
    else:
        file_path = os.path.join(file_dir, pt_name)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if not os.path.exists(file_path):
            pt_path = hf_hub_download(
                repo_id=repo_id,
                filename=pt_name,
                local_dir=file_dir,
            )
        else:
            pt_path=get_instance_path(file_path)
        return pt_path

def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def pil2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = pil2narry(value)
        list_in[i] = modified_value
    return list_in

def get_video_img(tensor):
    outputs = []
    for x in tensor:
        x = tensor_to_pil(x)
        outputs.append(x)
    return outputs


def nomarl_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_pil(samples)
    return img_pil

def get_local_path(comfy_file_path, model_path):
    path = os.path.join(comfy_file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform == 'win32':
        model_path = model_path.replace('\\', "/")
    return model_path

def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path

paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "model_index.json" in files:
                paths.append(os.path.relpath(root, start=search_path))

if paths:
    paths = ["none"] + [x for x in paths if x]
else:
    paths = ["none", ]
    
def instance_path(path, repo):
    if repo == "":
        if path == "none":
            repo = "none"
        else:
            model_path = get_local_path(comfy_file_path, path)
            repo = get_instance_path(model_path)
    return repo


def motion_sync_main(vis,width, height,driver_video,image,audio_form_video):
    # vis = FaceMeshVisualizer(draw_iris=False, draw_mouse=True, draw_eye=True, draw_nose=True, draw_eyebrow=True,
    #                          draw_pupil=True)
    
    imsize = (width, height)

    driver_video=os.path.join(folder_paths.input_directory,driver_video)
    image_name = "temp_" + ''.join(random.choice("0123456789") for _ in range(5))
    # base origin video
    if audio_form_video:
        audio_path=os.path.join(folder_paths.input_directory,f"{image_name}_audio.wav")
        video_clip = VideoFileClip(driver_video)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)
        video_clip.close()
        audio_clip.close()
    else:
        audio_path=None
    
    face_img = np.array(image)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    
   
    ref_image = os.path.join(tensorrt_lite, f"{image_name}.png")
    image.save(ref_image)
    
    lmk_extractor = LMKExtractor()
    
    input_frames_cv2 = [cv2.resize(center_crop_cv2(pil_to_cv2(i)), imsize) for i in pils_from_video(driver_video)]
    ref_frame = cv2.resize(face_img, (width, height))
    ref_det = lmk_extractor(ref_frame)
    # print(ref_det)
    
    sequence_driver_det = []
    try:
        for frame in input_frames_cv2:
            result = lmk_extractor(frame)
            assert result is not None, "{}, bad video, face not detected".format(driver_video)
            sequence_driver_det.append(result)
    except:
        print("face detection failed")
        exit()
    print(len(sequence_driver_det))
    
    if vis:
        pose_frames_driver = [vis.draw_landmarks((width, height), i["lmks"], normed=True) for i in sequence_driver_det]
        poses_add_driver = [(i * 0.5 + j * 0.5).clip(0, 255).astype(np.uint8) for i, j in
                            zip(input_frames_cv2, pose_frames_driver)]
    
    save_dir=os.path.join(tensorrt_lite,image_name)
    os.makedirs(save_dir, exist_ok=True)
    
    sequence_det_ms = motion_sync(sequence_driver_det, ref_det)
    for i in range(len(sequence_det_ms)):
        with open('{}/{}.pkl'.format(save_dir, i), 'wb') as file:
            pickle.dump(sequence_det_ms[i], file)
    # if vis:
    #     pose_frames = [vis.draw_landmarks((width, height), i, normed=False) for i in sequence_det_ms]
    #     poses_add = [(i * 0.5 + ref_frame * 0.5).clip(0, 255).astype(np.uint8) for i in pose_frames]
    #
    # poses_cat = [np.concatenate([i, j], axis=1) for i, j in zip(poses_add_driver, poses_add)]
    # output_video_path = os.path.join(folder_paths.output_directory, f"{image_name}_.mp4")
    # if save_video:
    #     save_video_from_cv2_list(poses_cat, output_video_path, fps=fps)

    return  save_dir,audio_path


class Echo_LoadModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae":("STRING", {"default": "stabilityai/sd-vae-ft-mse"}),
                "denoising":("BOOLEAN", {"default": True},),
                "pose_mode": (["none","normal", "turbo"],),
                "draw_mouse": ("BOOLEAN", {"default": False},),
                "motion_sync": ("BOOLEAN", {"default": False},),
            }
        }

    RETURN_TYPES = ("MODEL","MODEL","MODEL",)
    RETURN_NAMES = ("model","face_detector","visualizer")
    FUNCTION = "main_loader"
    CATEGORY = "EchoMimic"

    def main_loader(self,vae,denoising,pose_mode,draw_mouse,motion_sync):
 
        ############# model_init started #############
        ## vae init
        vae = AutoencoderKL.from_pretrained(vae).to("cuda", dtype=weight_dtype)
        ## reference net init
        pretrained_base_model_path=get_instance_path(weigths_current_path)
        
        #pre models
        download_weights(weigths_current_path,"lambdalabs/sd-image-variations-diffusers",subfolder="unet",pt_name="diffusion_pytorch_model.bin")
        download_weights(weigths_current_path,"lambdalabs/sd-image-variations-diffusers",subfolder="unet",pt_name="config.json")
        audio_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic", subfolder="audio_processor",
                                    pt_name="whisper_tiny.pt")
        
        if pose_mode=="normal":
            re_ckpt = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="reference_unet_pose.pth")
            motion_path = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="motion_module_pose.pth")
            denois_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="denoising_unet_pose.pth")
            face_locator_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="face_locator_pose.pth")
        elif pose_mode=="turbo":
            re_ckpt = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="reference_unet_pose.pth")
            motion_path = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                           pt_name="motion_module_pose_acc.pth")
            denois_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="denoising_unet_pose_acc.pth")
            face_locator_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                               pt_name="face_locator_pose.pth")
        else:
            re_ckpt = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="reference_unet.pth")
            motion_path = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="motion_module.pth")
            denois_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="denoising_unet.pth")
            face_locator_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="face_locator.pth")
        
       
        reference_unet = UNet2DConditionModel.from_config(
            pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device=device)

        reference_unet.load_state_dict(torch.load(re_ckpt, map_location="cpu"))
        
        ## denoising net init
        if denoising:
            if os.path.exists(motion_path): ### stage1 + stage2
                denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
                    pretrained_base_model_path,
                    motion_path,
                    subfolder="unet",
                    unet_additional_kwargs=infer_config.unet_additional_kwargs,
                ).to(dtype=weight_dtype, device=device)
            else:
                denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
                    pretrained_base_model_path,
                    "",
                    subfolder="unet",
                    unet_additional_kwargs={
                        "use_motion_module": False,
                        "unet_use_temporal_attention": False,
                        "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
                    }
                ).to(dtype=weight_dtype, device=device)
        else:
            ### only stage1
            denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
                pretrained_base_model_path,
                "",
                subfolder="unet",
                unet_additional_kwargs={
                    "use_motion_module": False,
                    "unet_use_temporal_attention": False,
                    "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
                }
            ).to(dtype=weight_dtype, device=device)
        
        denoising_unet.load_state_dict(torch.load(denois_pt, map_location="cpu"), strict=False)
        if pose_mode!="none":
            # face locator init
            face_locator = FaceLocator(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
                dtype=weight_dtype, device="cuda")
            face_locator.load_state_dict(torch.load(face_locator_pt))
            visualizer = FaceMeshVisualizer(draw_iris=False, draw_mouse=draw_mouse)
            if motion_sync:
                visualizer = FaceMeshVisualizer(draw_iris=False, draw_mouse=True, draw_eye=True, draw_nose=True, draw_eyebrow=True, draw_pupil=True)
        else:
            # face locator init
            face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
                dtype=weight_dtype, device="cuda")
            face_locator.load_state_dict(torch.load(face_locator_pt))
            visualizer = None
        
        ## load audio processor params
        audio_processor = load_audio_model(model_path=audio_pt, device=device)
        
        ## load face detector params
        face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709,
                              post_process=True, device=device)
        
        ############# model_init finished #############
    
        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)
        if pose_mode=="normal":
            pipe = AudioPose2VideoPipeline(
                vae=vae,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                audio_guider=audio_processor,
                face_locator=face_locator,
                scheduler=scheduler,
            ).to("cuda", dtype=weight_dtype)
        elif pose_mode=="turbo":
            pipe = AudioPose2VideoaccPipeline(
                vae=vae,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                audio_guider=audio_processor,
                face_locator=face_locator,
                scheduler=scheduler,
            ).to("cuda", dtype=weight_dtype)
        else:
            pipe = Audio2VideoPipeline(
                vae=vae,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                audio_guider=audio_processor,
                face_locator=face_locator,
                scheduler=scheduler,
            ).to("cuda", dtype=weight_dtype)
       
        return (pipe,face_detector,visualizer)
    

class Echo_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        input_path = folder_paths.get_input_directory()
        video_files = [f for f in os.listdir(input_path) if
                       os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ['webm', 'mp4', 'mkv', 'gif']]
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "pipe": ("MODEL",),
                "face_detector": ("MODEL",),
                "video_files": (["none"] + video_files,),
                "pose_dir":(pose_path_list_,),
                "seeds": ("INT", {"default": 0, "min": 0, "max":10000}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "sample_rate":  ("INT", {"default": 16000, "min": 8000, "max": 48000,"step": 1000,}),
                "facemask_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "facecrop_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "context_frames": ("INT", {"default": 12, "min": 0, "max": 50}),
                "context_overlap": ("INT", {"default": 3, "min": 0, "max": 10}),
                "length": ("INT", {"default": 120, "min": 100, "max": 5000, "step": 1, "display": "number"}),
                "width": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64, "display": "number"}),
                "audio_form_video": ("BOOLEAN", {"default": False},),
                "save_video": ("BOOLEAN", {"default": False},), },
            "optional": {
                "visualizer": ("MODEL",),}
        }
    
    RETURN_TYPES = ("IMAGE","AUDIO","FLOAT")
    RETURN_NAMES = ("image","audio","frame_rate")
    FUNCTION = "em_main"
    CATEGORY = "EchoMimic"
    
    def em_main(self, image,audio,pipe,face_detector,video_files,pose_dir,seeds,cfg, steps,fps,sample_rate,facemask_ratio,facecrop_ratio,context_frames,context_overlap,length,
                    width, height,audio_form_video,save_video,**kwargs):
        image=nomarl_upscale(image,width, height)
        #image=tensor_to_pil(image)
        visualizer = kwargs.get("visualizer")
        audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(5))
        audio_file = os.path.join(folder_paths.input_directory, f"{audio_file_prefix}_temp.wav")
        torchaudio.save(audio_file, audio["waveform"].squeeze(0), sample_rate=audio["sample_rate"])
        
        if visualizer: #pose
            if pose_dir=="none": #motion sync
                if video_files!="none":
                    pose_dir, audio_from_v = motion_sync_main(visualizer, width, height, video_files, image,
                                                              audio_form_video)
                    if audio_form_video:
                        audio_file = audio_from_v
                        waveform, sample_rate = torchaudio.load(audio_from_v)
                        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
                else:
                    pose_dir = os.path.join(current_path, "assets", "test_pose_demo_pose")  # default
            else:
                pose_dir = get_instance_path(os.path.join(tensorrt_lite, pose_dir))
              
        output_video = process_video(image, audio_file, width, height, length, seeds, facemask_ratio,
                                     facecrop_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps,
                                     device, pipe, face_detector, save_video,pose_dir, visualizer)
        gen = narry_list(output_video)  # pil列表排序
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
        frame_rate=float(fps)
        return (images,audio,frame_rate)


NODE_CLASS_MAPPINGS = {
    "Echo_LoadModel":Echo_LoadModel,
    "Echo_Sampler": Echo_Sampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Echo_LoadModel":"Echo_LoadModel",
    "Echo_Sampler": "Echo_Sampler",
}
