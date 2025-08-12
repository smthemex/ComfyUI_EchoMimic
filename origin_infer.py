# !/usr/bin/env python
# -*- coding: UTF-8 -*-

from diffusers import AutoencoderKL, DDIMScheduler
from safetensors.torch import load_file
import gc   
import torch
import os
from omegaconf import OmegaConf
import cv2
from PIL import Image
import numpy as np
try:
    from moviepy.editor import  VideoFileClip, AudioFileClip
except:
    try:
        from moviepy import VideoFileClip, AudioFileClip
    except:
        from moviepy import *
import datetime
import pickle
import shutil
from .utils import weight_dtype,download_weights, select_face,crop_and_pad,crop_and_pad_rectangle,center_resize_pad,cv2tensor,tensor_upscale,tensor2cv,motion_sync_main,img_padding,estimate_ratio,align_img,affine_img,get_video_pose,get_pose_params,save_pose_params,draw_pose_select_v2
from .src.models.whisper.audio2feature import load_audio_model
import folder_paths
MAX_SIZE = 768

def gc_clear():  # 释放显存
    torch.cuda.empty_cache()
    gc.collect()


def load_vae(vae_path,device,current_path):
    vae_state_dict=load_file(folder_paths.get_full_path("vae", vae_path))
    vae_config = AutoencoderKL.load_config(os.path.join(current_path, "configs/config.json"))
    vae = AutoencoderKL.from_config(vae_config).to(device, weight_dtype)
    vae.load_state_dict(vae_state_dict, strict=False)
    del vae_state_dict
    gc.collect()
    torch.cuda.empty_cache()
    return vae

def pre_model(weigths_current_path, version, infer_mode):

    if version in ["V1", "V2"]:
        download_weights(weigths_current_path, "lambdalabs/sd-image-variations-diffusers", subfolder="unet",
                        pt_name="diffusion_pytorch_model.bin")
        download_weights(weigths_current_path, "lambdalabs/sd-image-variations-diffusers", subfolder="unet",
                        pt_name="config.json")
        audio_pt = download_weights(weigths_current_path, "BadToBest/EchoMimic", subfolder="audio_processor",
                                    pt_name="whisper_tiny.pt")
        pose_encoder_pt,face_locator_pt=  None,None
        if version == "V1":
            print("****** refer in EchoMimic V1 mode!******")
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
                print("****** refer in EchoMimic V2 normal mode!******")
                motion_path = download_weights(weigths_current_path_v2, "BadToBest/EchoMimicV2",
                                               pt_name="motion_module.pth")
                denois_pt = download_weights(weigths_current_path_v2, "BadToBest/EchoMimicV2",
                                             pt_name="denoising_unet.pth")
            else: #pose_acc
                print("****** refer in EchoMimic V2 acc mode!******")
                motion_path = download_weights(weigths_current_path_v2, "BadToBest/EchoMimicV2",
                                               pt_name="motion_module_acc.pth")
                denois_pt = download_weights(weigths_current_path_v2, "BadToBest/EchoMimicV2",
                                             pt_name="denoising_unet_acc.pth")
        return re_ckpt, audio_pt, face_locator_pt, motion_path, denois_pt,pose_encoder_pt
    else: # V3
        download_weights(weigths_current_path, "BadToBest/EchoMimicV3", subfolder="transformer",
                        pt_name="diffusion_pytorch_model.safetensors")
        download_weights(weigths_current_path, "BadToBest/EchoMimicV3", subfolder="transformer",
                         
                        pt_name="config.json")
        return True


def Echo_v1_load_model(vae_path,weigths_current_path,version, infer_mode,denoising,current_path,device,lowvram):
    from .src.models.unet_2d_condition import UNet2DConditionModel
    from .src.models.unet_3d_echo import EchoUNet3DConditionModel
    from .src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
    from .src.pipelines.pipeline_echo_mimic_acc import Audio2VideoPipeline as Audio2VideoACCPipeline
    from .src.pipelines.pipeline_echo_mimic_pose import AudioPose2VideoPipeline
    from .src.pipelines.pipeline_echo_mimic_pose_acc import AudioPose2VideoPipeline as AudioPose2VideoaccPipeline

    infer_config = OmegaConf.load(os.path.join(current_path, "configs", "inference", "inference_v2.yaml"))
    print("****** refer in EchoMimic V1 mode!******")
    re_ckpt, audio_pt, face_locator_pt, motion_path, denois_pt,pose_encoder_pt=pre_model(weigths_current_path, version, infer_mode)
    # unet init
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

    re_state = torch.load(re_ckpt, map_location="cpu")
    reference_unet.load_state_dict(re_state, strict=False)
    del re_state
    gc_clear()

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
    denoising_state = torch.load(denois_pt, map_location="cpu")
    denoising_unet.load_state_dict(denoising_state, strict=False)
    del denoising_state
    gc_clear()

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    vae=load_vae(vae_path,device,current_path)

    if infer_mode == "pose_normal_dwpose" or  infer_mode == "pose_normal_sapiens":
        pipe = AudioPose2VideoPipeline(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            # audio_guider=audio_processor,
            # face_locator=face_locator,
            scheduler=scheduler,
        ).to(dtype=weight_dtype)
    elif infer_mode == "pose_acc":
        pipe = AudioPose2VideoaccPipeline(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            #audio_guider=audio_processor,
            #face_locator=face_locator,
            scheduler=scheduler,
        ).to(dtype=weight_dtype)
    elif infer_mode == "audio_drived":
        pipe = Audio2VideoPipeline(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            # audio_guider=audio_processor,
            # face_locator=face_locator,
            scheduler=scheduler,
        ).to(dtype=weight_dtype)
    else: #audio_drived_acc
        pipe = Audio2VideoACCPipeline(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            # audio_guider=audio_processor,
            # face_locator=face_locator,
            scheduler=scheduler,
        ).to(dtype=weight_dtype)

    pipe.enable_vae_slicing()
    if lowvram:
        pipe.enable_sequential_cpu_offload()
   
    return pipe,audio_pt, face_locator_pt


def Echo_v2_load_model(vae_path,weigths_current_path, version, infer_mode,current_path,device,lowvram):
    from .echomimic_v2.src.models.unet_2d_condition import UNet2DConditionModel as UNet2DConditionModelV2
    from .echomimic_v2.src.models.unet_3d_emo import  EMOUNet3DConditionModel as EMOUNet3DConditionModelV2
    
    from .echomimic_v2.src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
    from .echomimic_v2.src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline as EchoMimicV2PipelineACC

    infer_config_v2 = OmegaConf.load(os.path.join(current_path, "echomimic_v2/configs/inference/inference_v2.yaml"))
    print("****** refer in EchoMimic V2 mode!******")
    re_ckpt, audio_pt, face_locator_pt, motion_path, denois_pt,pose_encoder_pt=pre_model(weigths_current_path, version, infer_mode)
    # unet init
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
    gc_clear()      

    denoising_unet = EMOUNet3DConditionModelV2.from_pretrained_2d(
        weigths_current_path,
        motion_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config_v2.unet_additional_kwargs,
        ).to(dtype=weight_dtype)
    denoising_state = torch.load(denois_pt, map_location="cpu")
    denoising_unet.load_state_dict(denoising_state, strict=False)
    del denoising_state
    gc_clear()

 
    sched_kwargs = OmegaConf.to_container(infer_config_v2.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)
    vae=load_vae(vae_path,device,current_path)

    if infer_mode != "pose_acc":
        pipe = EchoMimicV2Pipeline(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            #audio_guider=audio_processor,
            #pose_encoder=pose_net,
            scheduler=scheduler, )
    else:
        pipe = EchoMimicV2PipelineACC(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            # audio_guider=audio_processor,
            # pose_encoder=pose_net,
            scheduler=scheduler, )

    pipe.enable_vae_slicing()
    if lowvram:
        pipe.enable_sequential_cpu_offload()
    return pipe, pose_encoder_pt, audio_pt




def Echo_v1_predata(face_img,audio_path,fps,audio_pt,face_locator_pt,device,infer_mode,draw_mouse,motion_sync_,lowvram,
                    width,height,facemask_dilation_ratio,facecrop_dilation_ratio,video_images,audio_file_prefix,cur_path,tensorrt_lite,length,pose_dir):
    #pre model
    from .src.models.face_locator import FaceLocator
    from facenet_pytorch import MTCNN



    audio_guider = load_audio_model(model_path=audio_pt, device=device)
    if  "pose" in infer_mode :
        from .src.utils.draw_utils import FaceMeshVisualizer
        # face locator init
        face_locator = FaceLocator(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
            dtype=weight_dtype, device=device)
        face_dict=torch.load(face_locator_pt,weights_only=False,)
        face_locator.load_state_dict(face_dict, strict=False)
        del face_dict
        gc_clear()
        if motion_sync_:
            visualizer = FaceMeshVisualizer(draw_iris=False, draw_mouse=True, draw_eye=True, draw_nose=True,
                                            draw_eyebrow=True, draw_pupil=True)
        else:
            visualizer = FaceMeshVisualizer(draw_iris=False, draw_mouse=draw_mouse)
    else:
        # face locator init
        face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
            dtype=weight_dtype, device=device)
        face_dict=torch.load(face_locator_pt,weights_only=False)
        face_locator.load_state_dict(face_dict, strict=False)
        visualizer = None
    del face_dict
    gc_clear()
    
    face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709,
                            post_process=True, device=device)
    
    # get data
    whisper_feature = audio_guider.audio2feat(audio_path)
    whisper_chunks = audio_guider.feature2chunks(feature_array=whisper_feature, fps=fps)
    

    #### face musk prepare
    face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')
    det_bboxes, probs = face_detector.detect(face_img)
    select_bbox = select_face(det_bboxes, probs)
    
    if select_bbox is None:
        face_mask[:, :] = 255
    else:
        xyxy = select_bbox[:4].astype(float)  # 面部处理出来是浮点数，无法实现整形
        xyxy = np.round(xyxy).astype("int")
        
        rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2] #56 227 132 268
        r_pad = int((re - rb) * facemask_dilation_ratio) # ratio：0.1 遮罩膨胀系数 17*2
        c_pad = int((ce - cb) * facemask_dilation_ratio) # ratio：0.1 遮罩膨胀系数 14*2
        face_mask[rb - r_pad: re + r_pad, cb - c_pad: ce + c_pad] = 255
       
        #### face crop ####
        if facecrop_dilation_ratio<1.0:
            if facecrop_dilation_ratio==0:
                facecrop_dilation_ratio=1
            r_pad_crop = int((re - rb) * facecrop_dilation_ratio)  # ratio 0.5  r_pad_crop：85,c_pad_crop:68
            c_pad_crop = int((ce - cb) * facecrop_dilation_ratio)  # ratio 1.0  r_pad_crop：171,c_pad_crop:136
            
            crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]),
                         min(re + r_pad_crop, face_img.shape[0])]
            
            if width == height:
                # 输出图片指定尺寸，如果是非方形，则会变形
                face_img_i, ori_face_rect_i = crop_and_pad(face_img, crop_rect)
                face_mask_m, ori_mask_rect_m = crop_and_pad(face_mask, crop_rect)  # (0, 7, 384, 391)
                face_img = cv2.resize(face_img_i, (width, height))
                face_mask = cv2.resize(face_mask_m, (width, height))
            else:
                face_img,face_mask=crop_and_pad_rectangle(face_img,face_mask,crop_rect)
                face_img= cv2tensor(face_img).permute(0, 2, 3, 1)#[1, 3, 357, 245] =>[[1,357, 245,3]]
                face_mask = cv2tensor(face_mask).permute(0, 2, 3, 1)
                face_img=tensor_upscale(face_img, width, height)
                face_img=tensor2cv(face_img)
                face_mask = tensor_upscale(face_mask, width, height)
                face_mask=cv2.cvtColor(tensor2cv(face_mask), cv2.COLOR_BGR2GRAY)#二值化
                ret, face_mask = cv2.threshold(face_mask, 0, 255, cv2.THRESH_BINARY)
                
        else: #when ratio=1 no crop
            print("when facecrop_ratio=1.0,The maximum image size will be obtained, but there may be edge deformation.** 选择最大裁切为1.0时，边缘可能会出现形变！")
        
    ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
    
    
    if visualizer:
        if pose_dir in ["pose_01","pose_02","pose_03","pose_04","pose_fight","pose_good","pose_salute","pose_ultraman"]:  # motion sync
            if isinstance(video_images,torch.Tensor):
                print("**** Use  video pose drive video! ****")
                pose_dir_path,video_len = motion_sync_main(visualizer, width, height, video_images, face_img,facecrop_dilation_ratio,
                                                          audio_file_prefix)
            else:
                print ("**** You need link video_images for drive video,but get none ,so use default  pkl driver  ****")
                pose_dir = os.path.join(cur_path, "assets", "test_pose_demo_pose")  # default
        else:
            print("**** Use  pkl drive video! ****")
            pose_dir_path = os.path.join(tensorrt_lite, pose_dir)
            files_and_directories = os.listdir(pose_dir_path)
            # 过滤出文件，排除子目录
            files = [f for f in files_and_directories if os.path.isfile(os.path.join(pose_dir_path, f))]
            video_len=len(files)
        if length>video_len:
            print(f"**** video length {video_len} is less than length,use {video_len} as {length}  ****")
            length=video_len
        pose_list = []
        for index in range(len(os.listdir(pose_dir_path))):
            tgt_musk_path = os.path.join(pose_dir_path, f"{index}.pkl")
            with open(tgt_musk_path, "rb") as f:
                tgt_kpts = pickle.load(f)
            tgt_musk = visualizer.draw_landmarks((width, height), tgt_kpts,facecrop_dilation_ratio)
            tgt_musk_pil = Image.fromarray(np.array(tgt_musk).astype(np.uint8)).convert('RGB')
            pose_list.append(
                torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device="cuda").permute(2, 0, 1) / 255.0)
        face_mask_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
    else:
        print("**** Use  audio drive video! ****")
        face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device="cuda").unsqueeze(0).unsqueeze(
            0).unsqueeze(0) / 255.0   
    mask_len=face_mask_tensor.shape[2]    
    face_locator_tensor = face_locator(face_mask_tensor)

    emb={"whisper_chunks":whisper_chunks,"face_locator_tensor":face_locator_tensor,"ref_image_pil":ref_image_pil,"length":length,"mask_len":mask_len,"audio_path":audio_path}
    return emb




def Echo_v2_predata(ref_image_pil,uploaded_audio,height,width,pose_encoder_pt,audio_pt,cur_path,video_images,tensorrt_lite,device,fps,length,infer_mode,weigths_current_path,pose_dir):
    from .echomimic_v2.src.models.pose_encoder import PoseEncoder

    pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(device=device,
                dtype=weight_dtype)
    pose_state = torch.load(pose_encoder_pt,map_location="cpu")
    pose_net.load_state_dict(pose_state)
    del pose_state
    gc_clear()

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
            pose_model_dir = pose_dir_bf16
        else:
            if os.path.exists(pose_dir_32):
                pose_model_dir = pose_dir_32
            else:
                pose_model_dir = ""
        visualizer = SapiensPoseEstimation(local_pose=pose_model_dir, model_dir=weigths_current_path, dtype=dtype)
    else:
        visualizer = None

    audio_processor = load_audio_model(model_path=audio_pt, device=device)
    whisper_feature = audio_processor.audio2feat(uploaded_audio)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)

    if infer_mode == "pose_normal_dwpose":
        face_detector ="dwpose"
    elif infer_mode == "pose_normal_sapiens":
        face_detector = "sapiens"
    else:
        face_detector = None


    # 处理输入图片的尺寸
    panding_img=img_padding(height, width, ref_image_pil) # 不管输出图片是何种尺寸，为保证图片质量，将输入图片转为为正方形，横裁切，竖填充，长宽为输出尺寸最大
    #### try input image Body alignment 暂时用sapiens
    # 将高宽改成最大图幅，方便裁切
    height = max(height, width)
    width = max(height, width)
    
    if visualizer and face_detector=="sapiens":
        visualizer.move_to_cuda()
        base_image=cv2.imread(os.path.join(cur_path,"echomimic_v2/assets/halfbody_demo/refimag/natural_bk_openhand/0222.png"))
        base_image=cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        
        _, base_image_key, base_image_box_xy = visualizer(np.asarray(base_image), None) #获取基准图片key 和xy数据 1024*1024
        base_image_length, base_image_left_eye_y = estimate_ratio(base_image_key, base_image_box_xy)
        
        panding_img_align = img_padding(1024, 1024, ref_image_pil) #裁切输入图片为1024*1024
        
        _, input_img_key, input_img_box_xy = visualizer(np.asarray(panding_img_align), None) #获取实际输入图片的key 和人体box数据
        input_img_length, input_img_left_eye_y = estimate_ratio(input_img_key, input_img_box_xy) #眼睛坐标为绝对值
    
        print(base_image_length,base_image_left_eye_y,input_img_length,input_img_left_eye_y) #603 [201] 679 [220]
        
        if base_image_length and base_image_left_eye_y and input_img_length and input_img_left_eye_y:
            if abs(base_image_length / 1024 - input_img_length / 1024) > 0.005:  # 比例不同须基于输入图片对齐
                print(
                    " *** Start input image align . 基于基准图片，开始输入图片的对齐! ***")
                input_img_left_eye_y_ = input_img_left_eye_y[0] #基于1024的绝对值
                base_image_left_eye_y_ = base_image_left_eye_y[0]#基于1024的绝对值
               
                panding_img=align_img(base_image_length, input_img_length, 1024, 1024, panding_img_align, base_image_left_eye_y_,
                          input_img_left_eye_y_)
                
            else:  # 人体比例接近，但是高度不对，也需要对齐
                print(
                    "Starting the input image shift based on the base image . 基于基准图片，开始输入图片手势平移对齐 ! ***")
                if abs(base_image_left_eye_y[0] / height - input_img_left_eye_y[
                    0] / height) > 0.005:
                    panding_img = affine_img(base_image_left_eye_y, input_img_left_eye_y, panding_img_align)
            print("input image Body alignment is done")
        panding_img=cv2.resize(panding_img, (width, height), interpolation=cv2.INTER_AREA) #基于1024做的对比，缩放回最大的输出尺寸
        if not isinstance(video_images,torch.Tensor):#非视频驱动时，完成对齐后，卸载dino模型
            visualizer.enable_model_cpu_offload()
            gc.collect()
            torch.cuda.empty_cache()
    infer_image_pil=Image.fromarray(cv2.cvtColor(panding_img,cv2.COLOR_BGR2RGB))
    
    if visualizer and isinstance(video_images,torch.Tensor):
        print("***** start infer video to npy files for drive pose ! ***** ")
        video_len, _, _, _ = video_images.size()
        if video_len < 50:
            raise "input video has not much frames for driver,change your input video!"
        else:
            tensor_list = list(torch.chunk(video_images, chunks=video_len))
            input_frames_cv2 = [img_padding(height,width,i) for i in tensor_list] #不管输出图片是何种尺寸，为保证图片质量，将输入视频转为为正方形，横裁切，竖填充，长宽为输出尺寸最大
            
        if face_detector=="sapiens":
            audio_file_prefix=f"{audio_file_prefix}_sapiens"
        pose_dir = os.path.join(tensorrt_lite, audio_file_prefix)
        if not os.path.exists(pose_dir):
            os.makedirs(pose_dir)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            pose_dir = os.path.join(tensorrt_lite, f"{audio_file_prefix}_{timestamp}")
            os.makedirs(pose_dir)
        
        #convert_fps(ori_video_path, ori_video_path_new)
        # extract pose
        if face_detector=="dwpose":
            detected_poses, height_video, width_video, ori_frames = get_video_pose(visualizer, input_frames_cv2,
                                                                                   max_frame=None)
            # parameterize pose
            res_params = get_pose_params(detected_poses, MAX_SIZE)
            # save pose to npy
            pose_dir = save_pose_params(detected_poses, res_params['pose_params'],
                                        res_params['draw_pose_params'], pose_dir)
            USE_Default=True
            print(f"All Finished,video frames number is {len(input_frames_cv2)}")
        else:
            # 首帧手势对齐输入图片，keypoint数据左眼，肩膀及手肘，
            _, first_key, first_box_xy = visualizer(np.asarray(input_frames_cv2[0]), None)
            if not first_box_xy:  # first frame maybe empty or no preson,skip it,try find sceond
                print("*********first frame don't has person,skip it**********")
                for i in range(len(input_frames_cv2)):
                    _, first_key, first_box_xy = visualizer(np.asarray(input_frames_cv2[i + 1]), None)
                    if first_box_xy:
                        break
            
            _, input_key, input_box_xy = visualizer(np.asarray(panding_img), None)
            first_length, first_left_eye_y = estimate_ratio(first_key, first_box_xy)
            input_length, input_left_eye_y = estimate_ratio(input_key, input_box_xy)
            # print(first_length,first_left_eye_y,input_length,input_left_eye_y) #160.0 [236.0] 158.0 [151.0] 眼睛高度为绝对值
            if first_length and first_left_eye_y and input_length and input_left_eye_y:
                if abs(input_length / height - first_length / height) > 0.005:  # 比例不同须基于输入图片对齐
                    print(
                        "Starting the first frame gesture alignment based on the input image *** 基于输入图片，开始首帧手势缩放对齐 !")
                    input_left_eye_y_ = input_left_eye_y[0]
                    first_left_eye_y_ = first_left_eye_y[0]
                    input_frames_cv2 = [align_img(input_length, first_length, height, width, i, input_left_eye_y_,
                                                  first_left_eye_y_) for i in input_frames_cv2]
                else:  # 人体比例接近，但是高度不对，也需要对齐
                    print(
                        "Starting the first frame shift based on the input image *** 基于输入图片，开始首帧手势平移对齐 !")
                    if abs(input_left_eye_y[0] / height - first_left_eye_y[
                        0] / height) > 0.005:
                        input_frames_cv2 = [affine_img(input_left_eye_y, first_left_eye_y, i) for i in input_frames_cv2]
            empty_index = []
            for i, img in enumerate(input_frames_cv2):
                pose_img, _, BOX_ = visualizer(np.asarray(img), [5])
                if not BOX_:
                    pose_img = np.zeros((width, height, 3), np.uint8)  # 防止空帧报错
                    empty_index.append(i)  # 记录空帧索引
                np.save(os.path.join(pose_dir, f"{i}"), pose_img)
                # cv2.imwrite(f"{i}.png", pose_img)
            
            if empty_index:
                print(
                    f"********* The index of frames list : {empty_index} , which is no person find in images *********")
                
                if len(empty_index) == 1:
                    if empty_index[0] != 0:
                        shutil.copy2(os.path.join(pose_dir, f"{empty_index[0]}.npy"),
                                     os.path.join(pose_dir, f"{empty_index[0] - 1}.npy"))  # 抽前帧覆盖
                    else:
                        shutil.copy2(os.path.join(pose_dir, f"{empty_index[0]}.npy"),
                                     os.path.join(pose_dir, f"{empty_index[0] + 1}.npy"))  # 抽前帧覆盖
                else:
                    if 0 not in empty_index:
                        for i in empty_index:
                            shutil.copy2(os.path.join(pose_dir, f"{i}.npy"),
                                         os.path.join(pose_dir, f"{empty_index[i] - 1}.npy"))  # 抽前帧覆盖
                    
                    else:
                        for i, x in enumerate(empty_index):  # 先抽连续帧最末尾的后一帧盖0帧
                            if empty_index[i] != x:  # [0,1,x]
                                shutil.copy2(os.path.join(pose_dir, f"{0}.npy"),
                                             os.path.join(pose_dir, f"{i}.npy"))
                                break
                            else:
                                pass
                        
                        for i, x in enumerate(empty_index):  # 其他帧抽前帧覆盖
                            if i != 0:
                                shutil.copy2(os.path.join(pose_dir, f"{x}.npy"),
                                             os.path.join(pose_dir, f"{empty_index[i] - 1}.npy"))  # 抽前帧覆盖
            
            USE_Default = False
            visualizer.enable_model_cpu_offload()
            gc.collect()
            torch.cuda.empty_cache()
    else:
        if pose_dir in ["pose_01","pose_02","pose_03","pose_04","pose_fight","pose_good","pose_salute","pose_ultraman"]:
            pose_d=pose_dir.split("_")[-1]
            print(f"use default pose {pose_dir} for running !")
            pose_dir = os.path.join(cur_path, f"echomimic_v2/assets/halfbody_demo/pose/{pose_d}")
            USE_Default = True
        else:
            print(
                "Use NPY files for custom videos, which must be located in directory 'comfyui/input/tensorrt_lite'")
            pose_dir = os.path.join(tensorrt_lite, pose_dir)
            USE_Default=False if "sapiens" in pose_dir else True
            
        
    
    #final_fps = fps
    start_idx = 0
    audio_clip = AudioFileClip(uploaded_audio)
    
    L = min(int(audio_clip.duration * fps),length,len(os.listdir(pose_dir))) # if above will cause error
    #L=min(length,L) #length is definitely
    print(f"***** infer length is {L}")
    
    pose_list = []
    for index in range(start_idx, start_idx + L):
        tgt_musk_path = os.path.join(pose_dir, "{}.npy".format(index))
        if USE_Default:
            detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
            imh_new, imw_new, rb, re, cb, ce = detected_pose[
                'draw_pose_params']  # print(imh_new, imw_new, rb, re, cb, ce) 官方示例蒙版的尺寸是768*768
            im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)  # 缩放比例为1，im也是768 ref_w！=768
            im = np.transpose(np.array(im), (1, 2, 0))
            tgt_musk = np.zeros((imw_new, imh_new, 3)).astype('uint8')
            tgt_musk[rb:re, cb:ce, :] = im
        else:
            tgt_musk = np.load(tgt_musk_path, allow_pickle=True)

        tgt_musk = center_resize_pad(tgt_musk, width, height) # 缩放裁剪遮罩，防止遮罩非正方形
        tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
        
        pose_list.append(
            torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device=device).permute(2, 0, 1) / 255.0)
    
    poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0).to(device)
    #print(f"poses_tensor:{poses_tensor.shape}")
    mask_len=poses_tensor.shape[2]
    face_locator_tensor=pose_net(poses_tensor[:, :, :L, ...]).to(device)
   
    emb ={"whisper_chunks":whisper_chunks,"infer_image_pil":infer_image_pil,"LEN":L,"mask_len":mask_len,"start_idx":start_idx,"face_locator_tensor": face_locator_tensor,"ref_image_pil":ref_image_pil,"H_change":height,"W_change":width}

    return  emb
