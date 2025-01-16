# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import io
import logging
import os
import random
import numpy as np
import torch
import torchaudio
import gc
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from .src.models.unet_2d_condition import UNet2DConditionModel
from .src.models.unet_3d_echo import EchoUNet3DConditionModel
from .src.models.whisper.audio2feature import load_audio_model
from .src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from .src.pipelines.pipeline_echo_mimic_acc import Audio2VideoPipeline as Audio2VideoACCPipeline
from .src.pipelines.pipeline_echo_mimic_pose import AudioPose2VideoPipeline
from .src.pipelines.pipeline_echo_mimic_pose_acc import AudioPose2VideoPipeline as AudioPose2VideoaccPipeline
from .src.models.face_locator import FaceLocator
from .src.utils.draw_utils import FaceMeshVisualizer
from .src.utils.motion_utils import motion_sync
from .utils import find_directories, download_weights,  \
    process_video, narry_list, weight_dtype, cf_tensor2cv,process_video_v2
from .echomimic_v2.src.models.pose_encoder import PoseEncoder
from .echomimic_v2.src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
from .echomimic_v2.src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline as EchoMimicV2PipelineACC
from .echomimic_v2.src.models.unet_2d_condition import UNet2DConditionModel as UNet2DConditionModelV2
from .echomimic_v2.src.models.unet_3d_emo import  EMOUNet3DConditionModel as EMOUNet3DConditionModelV2
import folder_paths
import platform
import subprocess

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
current_path = os.path.dirname(os.path.abspath(__file__))

inference_config_path = os.path.join(current_path, "configs", "inference", "inference_v2.yaml")
infer_config = OmegaConf.load(inference_config_path)
inference_config_path_v2 = os.path.join(current_path, "echomimic_v2/configs/inference/inference_v2.yaml")
infer_config_v2 = OmegaConf.load(inference_config_path_v2)


# pre dir
weigths_current_path = os.path.join(folder_paths.models_dir, "echo_mimic")
if not os.path.exists(weigths_current_path):
    os.makedirs(weigths_current_path)

weigths_uet_current_path = os.path.join(weigths_current_path, "unet")
if not os.path.exists(weigths_uet_current_path):
    os.makedirs(weigths_uet_current_path)

weigths_vae_current_path = os.path.join(weigths_current_path, "vae")
if not os.path.exists(weigths_vae_current_path):
    os.makedirs(weigths_vae_current_path)

weigths_au_current_path = os.path.join(weigths_current_path, "audio_processor")
if not os.path.exists(weigths_au_current_path):
    os.makedirs(weigths_au_current_path)

tensorrt_lite = os.path.join(folder_paths.get_input_directory(), "tensorrt_lite")
if not os.path.exists(tensorrt_lite):
    os.makedirs(tensorrt_lite)


# ffmpeg
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


# *****************mian***************

class Echo_LoadModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("STRING", {"default": "stabilityai/sd-vae-ft-mse"}),
                "denoising": ("BOOLEAN", {"default": True},),
                "infer_mode": (["audio_drived", "audio_drived_acc", "pose_normal_dwpose","pose_normal_sapiens", "pose_acc"],),
                "draw_mouse": ("BOOLEAN", {"default": False},),
                "motion_sync": ("BOOLEAN", {"default": False},),
                "lowvram": ("BOOLEAN", {"default": False},),
                "version": (["V2", "V1", ],),
            }
        }
    
    RETURN_TYPES = ("MODEL_PIPE_E", "MODEL_FACE_E", "MODEL_VISUAL_E",)
    RETURN_NAMES = ("model", "face_detector", "visualizer",)
    FUNCTION = "main_loader"
    CATEGORY = "EchoMimic"
    
    def main_loader(self, vae, denoising, infer_mode, draw_mouse, motion_sync, lowvram, version):
        
        ############# model_init started #############
        
        ## vae init  #using local vae first
        try:
            vae = AutoencoderKL.from_pretrained(weigths_vae_current_path).to(device,dtype=weight_dtype)  # using local vae first
        except:
            try:
                vae = AutoencoderKL.from_pretrained(vae).to(device, dtype=weight_dtype) #use input
            except:
                try:
                    # try downlaod model ,and load local vae
                    download_weights(weigths_vae_current_path, "stabilityai/sd-vae-ft-mse", subfolder="",
                                     pt_name="diffusion_pytorch_model.safetensors")
                    download_weights(weigths_vae_current_path, "stabilityai/sd-vae-ft-mse", subfolder="",
                                     pt_name="config.json")
                    vae = AutoencoderKL.from_pretrained(weigths_vae_current_path).to(device, dtype=weight_dtype)
                except:
                    # if no model path,use default.
                    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=weight_dtype)
              
        ## reference net init
        #pretrained_base_model_path = get_instance_path(weigths_current_path)
        
        # pre base models
        download_weights(weigths_current_path, "lambdalabs/sd-image-variations-diffusers", subfolder="unet",
                         pt_name="diffusion_pytorch_model.bin")
        download_weights(weigths_current_path, "lambdalabs/sd-image-variations-diffusers", subfolder="unet",
                         pt_name="config.json")
        audio_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic", subfolder="audio_processor",
                                    pt_name="whisper_tiny.pt")
        
        # pre pth
        if version == "V1":
            logging.info("****** refer in EchoMimic V1 mode!******")
            if infer_mode == "pose_normal_dwpose" or  infer_mode == "pose_normal_sapiens" :
                re_ckpt = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                           pt_name="reference_unet_pose.pth")
                face_locator_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                                   pt_name="face_locator_pose.pth")
                motion_path = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                               pt_name="motion_module_pose.pth")
                denois_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                             pt_name="denoising_unet_pose.pth")
            
            elif infer_mode == "pose_acc":
                re_ckpt = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                           pt_name="reference_unet_pose.pth")
                motion_path = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                               pt_name="motion_module_pose_acc.pth")
                denois_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                             pt_name="denoising_unet_pose_acc.pth")
                face_locator_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                                   pt_name="face_locator_pose.pth")
            elif infer_mode == "audio_drived":
                re_ckpt = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="reference_unet.pth")
                face_locator_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                                   pt_name="face_locator.pth")
                motion_path = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="motion_module.pth")
                denois_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="denoising_unet.pth")
            else:
                re_ckpt = download_weights(weigths_current_path, "BadToBest/EchoMimic", pt_name="reference_unet.pth")
                face_locator_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                                   pt_name="face_locator.pth")
                motion_path = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                               pt_name="motion_module_acc.pth")
                denois_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic",
                                             pt_name="denoising_unet_acc.pth")
        else:
            
            weigths_current_path_v2 = os.path.join(weigths_current_path, "v2")
            if not os.path.exists(weigths_current_path_v2):
                os.makedirs(weigths_current_path_v2)
            re_ckpt = download_weights(weigths_current_path_v2, "BadToBest/EchoMimicV2",
                                       pt_name="reference_unet.pth")
            pose_encoder_pt = download_weights(weigths_current_path_v2, "BadToBest/EchoMimicV2",
                                               pt_name="pose_encoder.pth")
            
            if infer_mode!="pose_acc":
                logging.info("****** refer in EchoMimic V2 normal mode!******")
                motion_path = download_weights(weigths_current_path_v2, "BadToBest/EchoMimicV2",
                                               pt_name="motion_module.pth")
                denois_pt = download_weights(weigths_current_path_v2, "BadToBest/EchoMimicV2",
                                             pt_name="denoising_unet.pth")
            else: #pose_acc
                logging.info("****** refer in EchoMimic V2 acc mode!******")
                motion_path = download_weights(weigths_current_path_v2, "BadToBest/EchoMimicV2",
                                               pt_name="motion_module_acc.pth")
                denois_pt = download_weights(weigths_current_path_v2, "BadToBest/EchoMimicV2",
                                             pt_name="denoising_unet_acc.pth")
        # unet init
        if version == "V1":
            try:
                reference_unet = UNet2DConditionModel.from_config(
                    weigths_current_path,
                    subfolder="unet",
                ).to(dtype=weight_dtype)
            except:
                reference_unet = UNet2DConditionModel.from_pretrained(
                    weigths_current_path,
                    subfolder="unet",
                ).to(dtype=weight_dtype)
        else:
            try:
                reference_unet = UNet2DConditionModelV2.from_config(
                    weigths_current_path,
                    subfolder="unet",
                ).to(dtype=weight_dtype)
            except:
                reference_unet = UNet2DConditionModelV2.from_pretrained(
                    weigths_current_path,
                    subfolder="unet",
                ).to(dtype=weight_dtype)
            
        re_state = torch.load(re_ckpt, map_location="cpu")
        reference_unet.load_state_dict(re_state, strict=False)
        del re_state
        gc.collect()
        torch.cuda.empty_cache()
        
        ## denoising net init
        if version == "V1":
            if denoising:
                if os.path.exists(motion_path):  ### stage1 + stage2
                    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
                        weigths_current_path,
                        motion_path,
                        subfolder="unet",
                        unet_additional_kwargs=infer_config.unet_additional_kwargs,
                    ).to(dtype=weight_dtype)
                else:
                    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
                        weigths_current_path,
                        "",
                        subfolder="unet",
                        unet_additional_kwargs={
                            "use_motion_module": False,
                            "unet_use_temporal_attention": False,
                            "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
                        }
                    ).to(dtype=weight_dtype)
            else:
                ### only stage1
                denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
                    weigths_current_path,
                    "",
                    subfolder="unet",
                    unet_additional_kwargs={
                        "use_motion_module": False,
                        "unet_use_temporal_attention": False,
                        "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
                    }
                ).to(dtype=weight_dtype, )
        else:  # v2
            denoising_unet = EMOUNet3DConditionModelV2.from_pretrained_2d(
                weigths_current_path,
                motion_path,
                subfolder="unet",
                unet_additional_kwargs=infer_config_v2.unet_additional_kwargs,
            ).to(dtype=weight_dtype)
        denoising_state = torch.load(denois_pt, map_location="cpu")
        denoising_unet.load_state_dict(denoising_state, strict=False)
        del denoising_state
        gc.collect()
        torch.cuda.empty_cache()
        
        if version == "V1":
            if  "pose" in infer_mode :
                # face locator init
                face_locator = FaceLocator(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
                    dtype=weight_dtype, device=device)
                face_locator.load_state_dict(torch.load(face_locator_pt), strict=False)
                if motion_sync:
                    visualizer = FaceMeshVisualizer(draw_iris=False, draw_mouse=True, draw_eye=True, draw_nose=True,
                                                    draw_eyebrow=True, draw_pupil=True)
                else:
                    visualizer = FaceMeshVisualizer(draw_iris=False, draw_mouse=draw_mouse)
            else:
                # face locator init
                face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
                    dtype=weight_dtype, device=device)
                face_locator.load_state_dict(torch.load(face_locator_pt), strict=False)
                visualizer = None
        else:  # v2
            # pose net init
            pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(device=device,
                dtype=weight_dtype)
            pose_state = torch.load(pose_encoder_pt,map_location="cpu")
            pose_net.load_state_dict(pose_state)
            del pose_state
            gc.collect()
            torch.cuda.empty_cache()
            if infer_mode == "pose_normal_dwpose":
                print("using DWpose drive pose")
                from .echomimic_v2.src.models.dwpose.dwpose_detector import DWposeDetector
                dw_ll=download_weights(weigths_current_path, "yzd-v/DWPose", subfolder="",
                                 pt_name="dw-ll_ucoco_384.onnx")
                yolox_l = download_weights(weigths_current_path, "yzd-v/DWPose", subfolder="",
                                         pt_name="yolox_l.onnx")
                visualizer = DWposeDetector(model_det=yolox_l,model_pose=dw_ll,device=device)
                
            elif infer_mode == "pose_normal_sapiens":
                print("using Sapiens drive pose")
                from .src.pose import SapiensPoseEstimation
                pose_dir_32 = os.path.join(weigths_current_path,
                                           "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2")
                pose_dir_bf16 = os.path.join(weigths_current_path,
                                             "sapiens_1b_goliath_best_goliath_AP_639_torchscript_bf16.pt2")
                dtype = torch.float32
                if os.path.exists(pose_dir_bf16):
                    dtype = torch.float16
                    pose_dir = pose_dir_bf16
                else:
                    if os.path.exists(pose_dir_32):
                        pose_dir = pose_dir_32
                    else:
                        pose_dir = ""
                visualizer = SapiensPoseEstimation(local_pose=pose_dir, model_dir=weigths_current_path, dtype=dtype)
            else:
                visualizer = None
            
        
        ## load audio processor params
        audio_processor = load_audio_model(model_path=audio_pt, device=device)
        
        ## load face detector params
        if version == "V1":
            from facenet_pytorch import MTCNN
            face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709,
                                  post_process=True, device=device)
        else:
            if infer_mode == "pose_normal_dwpose":
                face_detector ="dwpose"
            elif infer_mode == "pose_normal_sapiens":
                face_detector = "sapiens"
            else:
                face_detector = None
            
        ############# model_init finished #############
        if version == "V1":
            sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        else:
            sched_kwargs = OmegaConf.to_container(infer_config_v2.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)
       
        if version == "V1":
            if infer_mode == "pose_normal_dwpose" or  infer_mode == "pose_normal_sapiens":
                pipe = AudioPose2VideoPipeline(
                    vae=vae,
                    reference_unet=reference_unet,
                    denoising_unet=denoising_unet,
                    audio_guider=audio_processor,
                    face_locator=face_locator,
                    scheduler=scheduler,
                ).to(dtype=weight_dtype)
            elif infer_mode == "pose_acc":
                pipe = AudioPose2VideoaccPipeline(
                    vae=vae,
                    reference_unet=reference_unet,
                    denoising_unet=denoising_unet,
                    audio_guider=audio_processor,
                    face_locator=face_locator,
                    scheduler=scheduler,
                ).to(dtype=weight_dtype)
            elif infer_mode == "audio_drived":
                pipe = Audio2VideoPipeline(
                    vae=vae,
                    reference_unet=reference_unet,
                    denoising_unet=denoising_unet,
                    audio_guider=audio_processor,
                    face_locator=face_locator,
                    scheduler=scheduler,
                ).to(dtype=weight_dtype)
            else: #audio_drived_acc
                pipe = Audio2VideoACCPipeline(
                    vae=vae,
                    reference_unet=reference_unet,
                    denoising_unet=denoising_unet,
                    audio_guider=audio_processor,
                    face_locator=face_locator,
                    scheduler=scheduler,
                ).to(dtype=weight_dtype)
        else:
            if infer_mode != "pose_acc":
                pipe = EchoMimicV2Pipeline(
                    vae=vae,
                    reference_unet=reference_unet,
                    denoising_unet=denoising_unet,
                    audio_guider=audio_processor,
                    pose_encoder=pose_net,
                    scheduler=scheduler, )
            else:
                pipe = EchoMimicV2PipelineACC(
                    vae=vae,
                    reference_unet=reference_unet,
                    denoising_unet=denoising_unet,
                    audio_guider=audio_processor,
                    pose_encoder=pose_net,
                    scheduler=scheduler, )
        pipe.enable_vae_slicing()
        if lowvram:
            pipe.enable_sequential_cpu_offload()
        model = {"pipe": pipe, "lowvram": lowvram,"version":version}
        return (model, face_detector, visualizer,)


class Echo_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        pose_path_list = ["pose_01","pose_02","pose_03","pose_04","pose_fight","pose_good","pose_salute","pose_ultraman"] + find_directories(tensorrt_lite) if find_directories(tensorrt_lite) else ["pose_01","pose_02","pose_03","pose_04","pose_fight","pose_good","pose_salute","pose_ultraman"]
        return {
            "required": {
                "image": ("IMAGE",),  # [B,H,W,C], C=3
                "audio": ("AUDIO",),
                "model": ("MODEL_PIPE_E",),
                "face_detector": ("MODEL_FACE_E",),
                "pose_dir": (pose_path_list,),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "fps": ("FLOAT", {"default": 25.0, "min": 5.0, "max": 120.0}),
                "sample_rate": ("INT", {"default": 16000, "min": 8000, "max": 48000, "step": 1000, }),
                "facemask_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "facecrop_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "context_frames": ("INT", {"default": 12, "min": 0, "max": 50}),
                "context_overlap": ("INT", {"default": 3, "min": 0, "max": 10}),
                "length": ("INT", {"default": 120, "min": 50, "max": 5000, "step": 1, "display": "number"}),
                "width": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64, "display": "number"}),
                "save_video": ("BOOLEAN", {"default": False},), },
            "optional": {
                "visualizer": ("MODEL_VISUAL_E",),
                "video_images": ("IMAGE",),  # [B,H,W,C], C=3,B>1
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("image", "audio", "frame_rate")
    FUNCTION = "em_main"
    CATEGORY = "EchoMimic"
    
    def em_main(self, image, audio, model, face_detector, pose_dir, seed, cfg, steps, fps, sample_rate, facemask_ratio,
                facecrop_ratio, context_frames, context_overlap, length,
                width, height, save_video, **kwargs):
        
        version= model.get("version")
        pipe = model.get("pipe")
        lowvram = model.get("lowvram")
        if not lowvram:
            pipe.to(device, torch.float16)
        
        image = cf_tensor2cv(image, width, height)  if version=="V1" else image  # v1 cv ,v2 tensor
        visualizer = kwargs.get("visualizer")
        video_images = kwargs.get("video_images")
        
        audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
        audio_file = os.path.join(folder_paths.get_input_directory(), f"audio_{audio_file_prefix}_temp.wav")
        
        # 减少音频数据传递导致的不必要文件存储
        buff = io.BytesIO()
        torchaudio.save(buff, audio["waveform"].squeeze(0), audio["sample_rate"], format="FLAC")
        with open(audio_file, 'wb') as f:
            f.write(buff.getbuffer())
        if version=="V1":
            output_video = process_video(image, audio_file, width, height, length, seed, facemask_ratio,
                                         facecrop_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps,
                                         pipe, face_detector, save_video, pose_dir, video_images, audio_file_prefix,
                                         visualizer)
           
        else:
            output_video=process_video_v2(image, audio_file, width, height, length, seed,
                             context_frames, context_overlap, cfg, steps, sample_rate, fps, pipe,
                             save_video, pose_dir, audio_file_prefix,visualizer,video_images,face_detector )
            
        gen = narry_list(output_video)  # pil列表排序
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
        frame_rate = float(fps)
        if not lowvram:  # for upsacle ,need  VR
            pipe.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return (images, audio, frame_rate)



NODE_CLASS_MAPPINGS = {
    "Echo_LoadModel": Echo_LoadModel,
    "Echo_Sampler": Echo_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Echo_LoadModel": "Echo_LoadModel",
    "Echo_Sampler": "Echo_Sampler",
}
