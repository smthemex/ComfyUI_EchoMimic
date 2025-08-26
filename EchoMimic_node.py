# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import io
import os
import random
import numpy as np
import torch
import torchaudio
import gc
import platform
import subprocess

import folder_paths

os.environ['DEEPFACE_HOME'] = os.path.join(folder_paths.models_dir,"echo_mimic")
print(os.path.join(folder_paths.models_dir,"echo_mimic"))

from .utils import find_directories,process_video, cf_tensor2cv,process_video_v2,load_images,nomarl_upscale
from .origin_infer import Echo_v1_load_model,Echo_v2_load_model,Echo_v1_predata,Echo_v2_predata
from .echomimic_v3.infer import load_v3_model,infer_v3,Config,Echo_v3_predata



MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
current_path = os.path.dirname(os.path.abspath(__file__))


# pre dir
weigths_current_path = os.path.join(folder_paths.models_dir, "echo_mimic")
if not os.path.exists(weigths_current_path):
    os.makedirs(weigths_current_path)

weigths_unet_current_path = os.path.join(weigths_current_path, "unet")
if not os.path.exists(weigths_unet_current_path):
    os.makedirs(weigths_unet_current_path)

weigths_vae_current_path = os.path.join(weigths_current_path, "vae")
if not os.path.exists(weigths_vae_current_path):
    os.makedirs(weigths_vae_current_path)

weigths_au_current_path = os.path.join(weigths_current_path, "audio_processor")
if not os.path.exists(weigths_au_current_path):
    os.makedirs(weigths_au_current_path)

tensorrt_lite = os.path.join(folder_paths.get_input_directory(), "tensorrt_lite")
if not os.path.exists(tensorrt_lite):
    os.makedirs(tensorrt_lite)

weigths_trans_current_path = os.path.join(weigths_current_path, "transformer")
if not os.path.exists(weigths_trans_current_path):
    os.makedirs(weigths_trans_current_path)

weigths_deepface_current_path = os.path.join(weigths_current_path, ".deepface/weights")
if not os.path.exists(weigths_deepface_current_path):
    os.makedirs(weigths_deepface_current_path)



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


# *****************main***************

class Echo_LoadModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": (folder_paths.get_filename_list("vae"),),
                "lora": (["None"]+folder_paths.get_filename_list("loras"),),
                "denoising": ("BOOLEAN", {"default": True},),
                "infer_mode": (["audio_drived", "audio_drived_acc", "pose_normal_dwpose","pose_normal_sapiens", "pose_acc"],),
                "lowvram": ("BOOLEAN", {"default": True},),
                "teacache_offload": ("BOOLEAN", {"default": True},),
                "use_mmgp": (["LowRAM_LowVRAM","None", "VerylowRAM_LowVRAM","LowRAM_HighVRAM","HighRAM_LowVRAM","HighRAM_HighVRAM" ],), 

                "version": (["V3","V2", "V1", ],), },
        }
    
    RETURN_TYPES = ("MODEL_PIPE_E", "MODEL_INFO_E")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "main_loader"
    CATEGORY = "EchoMimic"
    
    def main_loader(self, vae, lora,denoising, infer_mode,  lowvram,teacache_offload,use_mmgp, version):
        
        config_v3,audio_pt,face_locator_pt,pose_encoder_pt,tokenizer,temporal_compression_ratio=None,None,None,None,None,None
        if "V1"==version :
            model, audio_pt, face_locator_pt= Echo_v1_load_model(vae,weigths_current_path,version, infer_mode, denoising, current_path, device,lowvram)
        elif "V2"==version :
            model,pose_encoder_pt, audio_pt= Echo_v2_load_model(vae,weigths_current_path, version, infer_mode,current_path,device,lowvram)
        else:#v3
            config_v3=Config()
            config_v3.config_path=os.path.join(current_path, "echomimic_v3/config/config.yaml")
            config_v3.teacache_offload=teacache_offload
            config_v3.quantize_transformer=lowvram
            lora_path=folder_paths.get_full_path("loras", lora) if lora!="None" else None

            model, temporal_compression_ratio,tokenizer=load_v3_model(current_path,weigths_current_path,config_v3, device,use_mmgp,vae,lora_path)
        print("##### model loaded #####")
        info = {"lowvram": lowvram,"version":version,"tokenizer":tokenizer,
                "infer_mode":infer_mode,"audio_pt":audio_pt, "face_locator_pt":face_locator_pt,"pose_encoder_pt":pose_encoder_pt,"temporal_compression_ratio":temporal_compression_ratio}
        info.update({"config":config_v3})
        return (model,info)


class Echo_Predata:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        pose_path_list = ["pose_01","pose_02","pose_03","pose_04","pose_fight","pose_good","pose_salute","pose_ultraman"] + find_directories(tensorrt_lite) if find_directories(tensorrt_lite) else ["pose_01","pose_02","pose_03","pose_04","pose_fight","pose_good","pose_salute","pose_ultraman"]
        return {
            "required": {
                "info": ("MODEL_INFO_E",),
                "image": ("IMAGE",),  # [B,H,W,C], C=3
                "audio": ("AUDIO",),
                "prompt": ("STRING", {"multiline": True,"default":" Gesture, Body, Face Expressions and Movements: Hands and Fingers Movements: The person's hands are not visible in the image, so there are no specific hand or finger movements to describe. Body Positions and Posture: The person is standing upright with a straight posture. Body Coverage in Frame: The person is fully visible from the waist up. Face Expressions Change: The person appears to have a neutral expression with a slight smile. Eyes Movement: The eyes are looking directly at the camera. Head Movement: The head is slightly tilted forward.  Overall Description: The character is standing in a front-facing shot, wearing a pink knitted vest over a white collared shirt and a white pleated skirt. The background appears to be a studio setting with soft lighting and some blurred elements that suggest a modern, clean environment. The person is adorned with pearl earrings, adding a touch of elegance to their appearance. The overall impression is one of a professional or casual presentation, possibly for a broadcast or a photoshoot."}),
                "negative_prompt" :("STRING", {"multiline": True,"default":" Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers. Unclear and blurry hands. 手部快速摆动, 手指频繁抽搐, 夸张手势, 重复机械性动作." }),
                "pose_dir": (pose_path_list,),
                "width": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64, "display": "number"}),   
                "fps": ("FLOAT", {"default": 25.0, "min": 5.0, "max": 120.0}),
                "facemask_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "facecrop_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "length": ("INT", {"default": 100, "min": 12, "max": 5000, "step": 1, "display": "number"}),
                "partial_video_length": ([33,65,97,113, 129,193],),
                "draw_mouse": ("BOOLEAN", {"default": False},),
                "motion_sync_": ("BOOLEAN", {"default": False},),
                },
            "optional": {
                "clip": ("CLIP",),
                "clip_vision": ("CLIP_VISION",), 
                "video_images": ("IMAGE",),  # [B,H,W,C], C=3,B>1
            }

        }
    
    RETURN_TYPES = ("MODEL_EMB_E", )
    RETURN_NAMES = ("emb",)
    FUNCTION = "main_loader"
    CATEGORY = "EchoMimic"
    
    def main_loader(self, info, image,audio,prompt,negative_prompt,pose_dir,width,height,fps,facemask_ratio,facecrop_ratio,length,partial_video_length,draw_mouse,motion_sync_,**kwargs):

        version=info.get("version", "V3")
        lowvram=info.get("lowvram", False)

        text_encoder=kwargs.get("clip", None)
        image_encoder=kwargs.get("clip_vision", None)
        video_images = kwargs.get("video_images")


        ip_mask_path=""
        # pre img
        if version=="V1" or version=="V2":       
            face_img = cf_tensor2cv(image, width, height)  if version=="V1" else image  # v1 cv ,v2 tensor
        else:
            face_img = nomarl_upscale(image, width, height)

        #pre audio
        audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
        audio_file = os.path.join(folder_paths.get_input_directory(), f"audio_{audio_file_prefix}_temp.wav")
        buff = io.BytesIO()
   
        torchaudio.save(buff, audio["waveform"].squeeze(0), audio["sample_rate"], format="FLAC")
        with open(audio_file, 'wb') as f:
            f.write(buff.getbuffer())

        # pre data
        if "V1"==version :
            emb = Echo_v1_predata(face_img,audio_file,fps,info.get("audio_pt"),info.get("face_locator_pt"),device,info.get("infer_mode"),draw_mouse,motion_sync_,lowvram,
                    width,height,facemask_ratio,facecrop_ratio,video_images,audio_file_prefix,current_path,tensorrt_lite,length,pose_dir)
        elif "V2"==version :
            emb = Echo_v2_predata(face_img,audio_file,height,width,info.get("pose_encoder_pt"),info.get("audio_pt"),
                                  current_path,video_images,tensorrt_lite,device,fps,length,info.get("infer_mode"),weigths_current_path,pose_dir)
        else:#v3
            config=info.get("config")
            config.fps=fps
            config.partial_video_length=int(partial_video_length)
            emb= Echo_v3_predata(image_encoder,text_encoder,info.get("tokenizer"),face_img,audio_file,ip_mask_path,config,device,info.get("temporal_compression_ratio"),prompt,negative_prompt,weigths_current_path,current_path,length)
            emb["config"]=config
        emb.update(info)
        emb.update({"audio_file_prefix":audio_file_prefix,"fps":fps,"height":height,"width":width})

        return (emb,)


class Echo_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_PIPE_E",),
                "emb": ("MODEL_EMB_E",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "sample_rate": ("INT", {"default": 16000, "min": 8000, "max": 48000, "step": 1000, }),
                "context_frames": ("INT", {"default": 12, "min": 0, "max": 50}),
                "context_overlap": ("INT", {"default": 3, "min": 0, "max": 10}),     
                "save_video": ("BOOLEAN", {"default": False},), },    
        }
    
    RETURN_TYPES = ("IMAGE",  "FLOAT")
    RETURN_NAMES = ("image",  "frame_rate")
    FUNCTION = "em_main"
    CATEGORY = "EchoMimic"
    
    def em_main(self,model, emb, seed, cfg, steps, sample_rate,context_frames, context_overlap,save_video,):
        
        version= emb.get("version")
        lowvram = emb.get("lowvram")

        if version=="V1":
            model.to(device)
            output_video = process_video(emb.get("ref_image_pil"), emb.get("audio_path"), emb.get("width"), emb.get("height"), emb.get("length"), seed, emb.get("face_locator_tensor"),
                                         context_frames, context_overlap, cfg, steps, sample_rate, emb.get("fps"), model,save_video,emb.get("mask_len"), emb.get("audio_file_prefix"),emb.get("whisper_chunks"))
        elif version=="V2":
            model.to(device)
            output_video=process_video_v2(emb.get("infer_image_pil"),emb.get("ref_image_pil"), emb.get("audio_path"),emb.get("face_locator_tensor"), emb.get("W_change"), emb.get("H_change"), emb.get("start_idx"), seed,
                   context_frames, context_overlap, cfg, steps, sample_rate, emb.get("fps"), model,emb.get("mask_len"),emb.get("height"),emb.get("width"),
                  save_video,  emb.get("audio_file_prefix"),emb.get("LEN"),emb.get("whisper_chunks") )    

        else: #v3
            config = emb.get("config")
            config.num_inference_steps=steps
            config.guidance_scale=cfg

            #config.sample_size=(emb.get("height"),emb.get("width"))
            output_video=infer_v3(model, config, device,emb.get("video_length"),emb.get("prompt_embeds"),emb.get("negative_prompt_embeds"),
             emb.get("temporal_compression_ratio"),seed,emb.get("partial_video_length"),emb.get("audio_embeds"),emb.get("ip_mask"),
             emb.get("sample_height"),emb.get("sample_width"),emb.get("clip_context"),
             emb.get("ref_image_pil"),emb.get("audio_file_prefix"))

        frame_rate = float(emb.get("fps"))
        if not lowvram and version!="V3":  # for upsacle ,need  VR
            model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return (load_images(output_video) , frame_rate)



NODE_CLASS_MAPPINGS = {
    "Echo_LoadModel": Echo_LoadModel,
    "Echo_Predata":Echo_Predata,
    "Echo_Sampler": Echo_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Echo_LoadModel": "Echo_LoadModel",
    "Echo_Predata": "Echo_Predata",
    "Echo_Sampler": "Echo_Sampler",
}
