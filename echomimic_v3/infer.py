# -*- coding: utf-8 -*-
# ==============================================================================
# arxive: https://arxiv.org/abs/2507.03905
# GitHUb: https://github.com/antgroup/echomimic_v3
# Project Page: https://antgroup.github.io/ai/echomimic_v3/
# ==============================================================================
import torchvision.transforms.functional as TF
import os
import math
import datetime
from functools import partial
import gc
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
import comfy.model_management
try:
    from moviepy.editor import  VideoFileClip, AudioFileClip
except:
    try:
        from moviepy import VideoFileClip, AudioFileClip
    except:
        from moviepy import *
import librosa
import folder_paths
# Custom modules
from diffusers import FlowMatchEulerDiscreteScheduler

#from .src.dist import set_multi_gpus_devices
from .src.wan_vae import AutoencoderKLWan
from .src.wan_image_encoder import  CLIPModel
from .src.wan_text_encoder import  WanT5EncoderModel
from .src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
from .src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline

from .src.utils import (
    filter_kwargs,
    get_image_to_video_latent3,
    save_videos_grid,
)
from .src.fm_solvers import FlowDPMSolverMultistepScheduler
from .src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .src.cache_utils import get_teacache_coefficients



# --------------------- Configuration ---------------------
class Config:
    def __init__(self):
        # General settings
        self.ulysses_degree = 1
        self.ring_degree = 1
        self.fsdp_dit = False

        # Pipeline parameters
        self.num_skip_start_steps = 5
        self.teacache_offload = False
        self.cfg_skip_ratio = 0
        self.enable_riflex = False
        self.riflex_k = 6

        # Paths
        self.config_path = "config/config.yaml"
        self.model_name = "models/Wan2.1-Fun-V1.1-1.3B-InP"
        self.transformer_path = "models/transformer/diffusion_pytorch_model.safetensors"
        self.vae_path = None

        # Sampler and audio settings
        self.sampler_name ="Flow_DPM++"
        self.audio_scale = 1.0
        self.enable_teacache = False
        self.teacache_threshold = 0.1
        self.shift = 5.0
        self.use_un_ip_mask = False

        # Inference parameters
        self.negative_prompt = "Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers. Unclear and blurry hands. 手部快速摆动, 手指频繁抽搐, 夸张手势, 重复机械性动作."#Unclear gestures, broken hands, more than five fingers on one hand, extra fingers, fused fingers. "# Strange body and strange trajectory. Distortion.  "

        self.partial_video_length = 113
        self.overlap_video_length = 8
        self.neg_scale = 1.5
        self.neg_steps = 2
        self.guidance_scale = 3.5 #4.0 ~ 6.0
        self.audio_guidance_scale = 3.0 #2.0 ~ 3.0
        self.use_dynamic_cfg = True
        self.use_dynamic_acfg = True
        self.seed = 43
        self.num_inference_steps = 20
        self.lora_weight = 1.0

        # Model settings
        self.weight_dtype = torch.bfloat16
        self.sample_size = [768, 768]
        self.fps = 25

        # Test data paths
        self.base_dir = "datasets/echomimicv3_demos/"
        self.test_name_list = [
            'guitar_woman_01','guitar_man_01',
            'demo_cartoon_03','demo_cartoon_04',
            '2025-07-14-1036','2025-07-14-1942',
            '2025-07-14-2371','2025-07-14-3927',
            '2025-07-14-4513','2025-07-14-6032',
            '2025-07-14-7113','2025-07-14-7335',
            ]

        self.wav2vec_model_dir = "models/wav2vec2-base-960h"
        self.save_path = "outputs"
        self.quantize_transformer=True



# --------------------- Helper Functions ---------------------
def load_wav2vec_models(wav2vec_model_dir):
    """Load Wav2Vec models for audio feature extraction."""
    processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_dir)
    model = Wav2Vec2Model.from_pretrained(wav2vec_model_dir).eval()
    model.requires_grad_(False)
    model.to("cpu")
    return processor, model


def extract_audio_features(audio_path, processor, model_):
    """Extract audio features using Wav2Vec."""
    sr = 16000
    audio_segment, sample_rate = librosa.load(audio_path, sr=sr)
    input_values = processor(audio_segment, sampling_rate=sample_rate, return_tensors="pt").input_values
    input_values=input_values.to("cpu")
    features = model_(input_values).last_hidden_state
    return features.squeeze(0)


def get_sample_size(image, default_size):
    """Calculate the sample size based on the input image dimensions."""
    width, height = image.size
    original_area = width * height
    default_area = default_size[0] * default_size[1]

    if default_area < original_area:
        ratio = math.sqrt(original_area / default_area)
        width = width / ratio // 16 * 16
        height = height / ratio // 16 * 16
    else:
        width = width // 16 * 16
        height = height // 16 * 16

    return int(height), int(width)


def get_ip_mask(coords):
    y1, y2, x1, x2, h, w = coords
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = (Y.unsqueeze(-1) >= y1) & (Y.unsqueeze(-1) < y2) & (X.unsqueeze(-1) >= x1) & (X.unsqueeze(-1) < x2)
    
    mask = mask.reshape(-1)
    return mask.float()

def get_file_path(base_dir, folder, test_name, extensions):
    """Helper function to find the file path with multiple extensions."""
    for ext in extensions:
        path = os.path.join(base_dir, folder, f"{test_name}.{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No file found for '{test_name}' in '{folder}' with extensions: {extensions}")


def load_v3_model(node_dir,weigths_current_path,config, device,use_mmgp,vae_path,lora_path):

     # Load configuration file
    cfg = OmegaConf.load(config.config_path)
    config.model_name=os.path.join(node_dir,"Wan2.1-Fun-V1.1-1.3B-InP")

    config.transformer_path=os.path.join(weigths_current_path,"diffusion_pytorch_model.safetensors")

    # Load models
    transformer = WanTransformerAudioMask3DModel.from_pretrained(
        os.path.join(weigths_current_path,"transformer"),
        transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
        #low_cpu_mem_usage=True if not config.fsdp_dit else False,
        torch_dtype=config.weight_dtype,
    )
    if config.transformer_path is not None:
        if config.transformer_path.endswith("safetensors"):
          from safetensors.torch import load_file, safe_open
          state_dict = load_file(config.transformer_path)
        else:
            state_dict = torch.load(config.transformer_path)
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    if lora_path:
        from .lora_adapter import WanLoraWrapper
        lora_wrapper = WanLoraWrapper(transformer)
        lora_name = lora_wrapper.load_lora(lora_path)
        lora_wrapper.apply_lora(lora_name, 1.0)
        transformer=lora_wrapper.model
        config.sampler_name="Flow_Unipc"


    vae_path_=folder_paths.get_full_path("vae", vae_path)
    vae = AutoencoderKLWan.from_pretrained(vae_path_,
       # os.path.join(config.model_name, cfg['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
    ).to(config.weight_dtype)
    

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.model_name, "google/umt5-xxl"))


    # clip_image_encoder = CLIPModel.from_pretrained(
    #     os.path.join(config.model_name, cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    # ).to(config.weight_dtype).eval()


    # Load scheduler
    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[config.sampler_name]
    scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, OmegaConf.to_container(cfg['scheduler_kwargs'])))

    # Create pipeline
    pipeline = WanFunInpaintAudioPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        #text_encoder=text_encoder,
        scheduler=scheduler,
        #clip_image_encoder=clip_image_encoder,
    )
   
    if config.enable_teacache:
        coefficients = get_teacache_coefficients(config.model_name)
        pipeline.transformer.enable_teacache(
            coefficients, config.num_inference_steps, config.teacache_threshold,
            num_skip_start_steps=config.num_skip_start_steps, offload=config.teacache_offload
        )

    
    if use_mmgp!="None":
        from mmgp import offload, profile_type
        pipeline.to("cpu")
        if use_mmgp=="VerylowRAM_LowVRAM":
            offload.profile(pipeline, profile_type.VerylowRAM_LowVRAM,quantizeTransformer=config.quantize_transformer)
        elif use_mmgp=="LowRAM_LowVRAM":  
            offload.profile(pipeline, profile_type.LowRAM_LowVRAM,quantizeTransformer=config.quantize_transformer)
        elif use_mmgp=="LowRAM_HighVRAM":
            offload.profile(pipeline, profile_type.LowRAM_HighVRAM,quantizeTransformer=config.quantize_transformer)
        elif use_mmgp=="HighRAM_LowVRAM":
            offload.profile(pipeline, profile_type.HighRAM_LowVRAM,quantizeTransformer=config.quantize_transformer)
        elif use_mmgp=="HighRAM_HighVRAM":
            offload.profile(pipeline, profile_type.HighRAM_HighVRAM,quantizeTransformer=config.quantize_transformer)
    else:
        pipeline.to(device)
    temporal_compression_ratio=pipeline.vae.config.temporal_compression_ratio
    #print(f"temporal_compression_ratio: {temporal_compression_ratio}") #4
   

   


    return pipeline,temporal_compression_ratio,tokenizer

def infer_v3(pipeline, config, device,video_length,prompt_embeds,negative_prompt_embeds,
             temporal_compression_ratio,seed,partial_video_length,audio_embeds,ip_mask,sample_height,
             sample_width,clip_context,ref_img,audio_file_prefix):

    generator = torch.Generator(device=device).manual_seed(seed)

    if 4 == config.num_inference_steps and config.sampler_name=="Flow_Unipc":
        print("Using LCM schedulers")
        lcm_config = {
        "infer_steps": 4,
        "target_video_length": 81,
        "target_height": 480,
        "target_width": 832,
        "self_attn_1_type": "flash_attn3",
        "cross_attn_1_type": "flash_attn3",
        "cross_attn_2_type": "flash_attn3",
        "seed": 442,
        "sample_guide_scale": 5,
        "denoising_step_list": [1000, 750, 500, 250],
        "sample_shift": 5,
        "enable_cfg": False,
        "cpu_offload": False,
            }
        
        config_ =OmegaConf.create(lcm_config)
        from .src.flow_match_lcm import WanStepDistillScheduler
        pipeline.scheduler = WanStepDistillScheduler(config_)
    
    # if config.enable_riflex:
    #     pipeline.transformer.enable_riflex(k = config.riflex_k, L_test = latent_frames)

    #Generate video in chunks
    init_frames = 0
    last_frames = init_frames + partial_video_length
    new_sample = None

    while init_frames < video_length:
        if last_frames >= video_length:
            partial_video_length  = video_length - init_frames
            partial_video_length  = (
                int((partial_video_length  - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1
                if video_length != 1 else 1
            )
            latent_frames = (partial_video_length  - 1) // temporal_compression_ratio + 1

            if partial_video_length  <= 0:
                break

        input_video, input_video_mask, _ = get_image_to_video_latent3(
            ref_img, None, video_length=partial_video_length , sample_size=[sample_height, sample_width]
        )

        partial_audio_embeds = audio_embeds[:, init_frames * 2 : (init_frames + partial_video_length ) * 2]
       

        sample = pipeline(
            None,
            num_frames            = partial_video_length ,
            negative_prompt       = None,
            audio_embeds          = partial_audio_embeds,
            audio_scale           = config.audio_scale,
            ip_mask               = ip_mask,
            use_un_ip_mask        = config.use_un_ip_mask,
            height                = sample_height,
            width                 = sample_width,
            generator             = generator,
            neg_scale             = config.neg_scale,
            neg_steps             = config.neg_steps,
            use_dynamic_cfg       = config.use_dynamic_cfg,
            use_dynamic_acfg      = config.use_dynamic_acfg,
            guidance_scale        = config.guidance_scale,
            audio_guidance_scale  = config.audio_guidance_scale,
            num_inference_steps   = config.num_inference_steps,
            video                 = input_video,
            mask_video            = input_video_mask,
            clip_image            = None,
            cfg_skip_ratio        = config.cfg_skip_ratio,
            shift                 = config.shift,
            clip_context          =clip_context,
            prompt_embeds         =prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        ).videos
        
        if init_frames != 0:
            mix_ratio = torch.from_numpy(
                np.array([float(i) / float(config.overlap_video_length) for i in range(config.overlap_video_length)], np.float32)
            ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            new_sample[:, :, -config.overlap_video_length:] = (
                new_sample[:, :, -config.overlap_video_length:] * (1 - mix_ratio) +
                sample[:, :, :config.overlap_video_length] * mix_ratio
            )
            new_sample = torch.cat([new_sample, sample[:, :, config.overlap_video_length:]], dim=2)
            sample = new_sample
        else:
            new_sample = sample

        if last_frames >= video_length:
            break


        ref_img = [
            Image.fromarray(
                (sample[0, :, i].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
            ) for i in range(-config.overlap_video_length, 0)
        ]

        init_frames += partial_video_length  - config.overlap_video_length
        last_frames = init_frames + partial_video_length 

    video_path = os.path.join(folder_paths.output_directory, f"{audio_file_prefix}_echo.mp4")
    # video_audio_path = os.path.join(folder_paths.output_directory, f"{audio_file_prefix}_echo.mp4")

    print ("final sample shape:",sample.shape)
    video_length = sample.shape[2]
    print ("final length:",video_length)

    pli_list=save_videos_grid(sample[:, :, :video_length], video_path, fps=config.fps)

    # video_clip = VideoFileClip(video_path)
    # audio_clip = audio_clip.subclipped(0, video_length / config.fps)
    # video_clip = video_clip.with_audio(audio_clip)
    # video_clip.write_videofile(video_audio_path, codec="libx264", audio_codec="aac", threads=2)

    # os.system(f"rm -rf {video_path}")
    return pli_list


def Echo_v3_predata(clip_image_encoder,text_encoder,tokenizer,face_img,audio_path,ip_mask_path,config,device,
                    temporal_compression_ratio,prompt,negative_prompt,weigths_current_path,current_path,target_video_length):
    
    # Extract audio features 

    config.wav2vec_model_dir= os.path.join(weigths_current_path,"wav2vec2-base-960h")
    wav2vec_processor, wav2vec_model = load_wav2vec_models(config.wav2vec_model_dir)

    audio_clip = AudioFileClip(audio_path,)
    audio_features = extract_audio_features(audio_path, wav2vec_processor, wav2vec_model)
    
    #print ("audio_features:",audio_features.shape) #torch.Size([502, 768])
    audio_duration_frames = int(audio_clip.duration * config.fps)
   
    if target_video_length < audio_duration_frames:
        # 如果目标长度小于音频长度，截取音频特征
        target_audio_features_length = target_video_length * 2  # 音频特征通常是视频帧数的2倍
        if audio_features.shape[0] > target_audio_features_length:
            audio_features = audio_features[:target_audio_features_length]
        video_length = target_video_length
    else:
        # 如果目标长度大于音频长度，使用音频实际长度
        video_length = audio_duration_frames
 
    audio_embeds = audio_features.unsqueeze(0).to(device=device, dtype=config.weight_dtype)
    print ("infer video_length:",video_length)

    if not os.path.exists(ip_mask_path):
        try:
            from .src.face_detect import get_mask_coord
            result = get_mask_coord(face_img)
            if result is None:
                print("Error: Face detection no face. Use a default mask path.")
                y1, y2, x1, x2, h_, w_ = np.load(os.path.join(current_path,"echomimic_v3/datasets/echomimicv3_demos/masks/demo_ch_woman_04.npy"))
            else:
                print("Done: Face detection is done.")
                y1, y2, x1, x2, h_, w_=result
        except:
            print("Error: Face detection failed. Use a default mask path.")
            y1, y2, x1, x2, h_, w_ = np.load(os.path.join(current_path,"echomimic_v3/datasets/echomimicv3_demos/masks/demo_ch_woman_04.npy"))
    else:
        y1, y2, x1, x2, h_, w_ = np.load(ip_mask_path)


    video_length = (
        int((video_length - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1
        if video_length != 1 else 1
    )
    latent_frames = (video_length - 1) // temporal_compression_ratio + 1

    # if config.enable_riflex: #TODO
    #     pipeline.transformer.enable_riflex(k = config.riflex_k, L_test = latent_frames)

    # Adjust sample size and create IP mask
    sample_height, sample_width = get_sample_size(face_img, config.sample_size)
    downratio = math.sqrt(sample_height * sample_width / h_ / w_)
    coords = (
        y1 * downratio // 16, y2 * downratio // 16,
        x1 * downratio // 16, x2 * downratio // 16,
        sample_height // 16, sample_width // 16,
    )
    ip_mask = get_ip_mask(coords).unsqueeze(0)
    ip_mask = torch.cat([ip_mask]*3).to(device=device, dtype=config.weight_dtype)

    partial_video_length = int((config.partial_video_length - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1 if video_length != 1 else 1
    #print ("partial_video_length:",partial_video_length)
    latent_frames = (partial_video_length - 1) // temporal_compression_ratio + 1

    # get clip image
    _, _, clip_image = get_image_to_video_latent3(face_img, None, video_length=partial_video_length, sample_size=[sample_height, sample_width])

    # video_length = init_frames + partial_video_length

    if clip_image is not None:
        clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device) #torch.Size([3, 1008, 576])
        
        #clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
        clip_image=clip_image.permute(1, 2, 0).unsqueeze(0) #comfy need [B,C,H,W]
        clip_dict=clip_image_encoder.encode_image(clip_image)
        clip_context =clip_dict["penultimate_hidden_states"].to(device, config.weight_dtype)
        #print(clip_dict["image_embeds"].shape,clip_dict["last_hidden_state"].shape,clip_dict["penultimate_hidden_states"].shape,) #torch.Size([1, 1024]) torch.Size([1, 257, 1280]) torch.Size([1, 257, 1280])
       
        #print(clip_context.shape) #torch.Size([1, 257, 1280])
    else:
        clip_image = Image.new("RGB", (512, 512), color=(0, 0, 0))  
        clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device) 
        #clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
        clip_image=clip_image.permute(1, 2, 0).unsqueeze(0)
        clip_context =clip_image_encoder.encode_image(clip_image)["penultimate_hidden_states"].to(device, config.weight_dtype)
        clip_context = torch.zeros_like(clip_context)
    
    clip_image_encoder.patcher.cleanup()

    gc.collect()
    prompt_embeds, negative_prompt_embeds=encode_prompt(text_encoder,tokenizer,prompt,negative_prompt,True,1,device=device,dtype=config.weight_dtype)

    emb={"audio_embeds":audio_embeds,"video_length":video_length,"clip_context":clip_context,"sample_height":sample_height,"sample_width":sample_width,
         "partial_video_length":partial_video_length,"ip_mask":ip_mask,
         "prompt_embeds":prompt_embeds,"negative_prompt_embeds":negative_prompt_embeds,
          "ref_image_pil":face_img,"latent_frames":latent_frames,}
    return emb



def encode_prompt(
        text_encoder,
        tokenizer,
        prompt,
        negative_prompt = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        max_sequence_length: int = 512,
        device= None,
        dtype= None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device 

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # prompt_embeds = self._get_t5_prompt_embeds(
            #     prompt=prompt,
            #     num_videos_per_prompt=num_videos_per_prompt,
            #     max_sequence_length=max_sequence_length,
            #     device=device,
            #     dtype=dtype,
            # )
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
       
            prompt_attention_mask = text_inputs.attention_mask

            prompt_embeds=cf_clip(prompt, text_encoder,prompt_attention_mask,device, dtype)[0]

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            text_inputs = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
       
            prompt_attention_mask = text_inputs.attention_mask
            # negative_prompt_embeds = self._get_t5_prompt_embeds(
            #     prompt=negative_prompt,
            #     num_videos_per_prompt=num_videos_per_prompt,
            #     max_sequence_length=max_sequence_length,
            #     device=device,
            #     dtype=dtype,
            # )
            negative_prompt_embeds=cf_clip(negative_prompt, text_encoder,prompt_attention_mask,device, dtype)[0]
        return prompt_embeds, negative_prompt_embeds


def cf_clip(txt_list, clip,prompt_attention_mask,device, dtype):
    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
    pos_cond_list = []
    for i in txt_list:
        tokens_p = clip.tokenize(i)
        output_p = clip.encode_from_tokens(tokens_p, return_dict=True)  # {"pooled_output":tensor}
        cond_p = output_p.pop("cond").to(device, dtype)
        #print(cond_p.shape) #torch.Size([1, 231, 768])
        positive=[u[:v] for u, v in zip(cond_p, seq_lens)]
        pos_cond_list.append(positive)
   
    return pos_cond_list
