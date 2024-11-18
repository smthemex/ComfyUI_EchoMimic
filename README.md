# ComfyUI_EchoMimic
You can using EchoMimic in comfyui

[EchoMimici](https://github.com/BadToBest/EchoMimic/tree/main)：Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning  

---

## Updates:
**2024/11/18**  
* Fix the bug of facial cropping, fix the bug of non square character deformation, and change the video driver to only use other video loading nodes such as VH or use the generated pkl model driver;   
* 修复面部裁切的bug，修复非正方形人物变形的bug，视频驱动改成只能用其他视频加载节点比如VH或者用已生成好的pkl模型驱动；
* ﻿The magnification factor of 'facecrop-ratio' is '1/facecrop-ratio'. If set to 0.5, the face will be magnified twice. It is recommended to adjust facecrop-ratio to a smaller value only when the proportion of faces in the reference image or driving video is very small,Do not cut when it is 1 or 0;     
* facecrop_ratio的放大系数为1/facecrop_ratio，如果设置为0.5，面部会得到2倍的放大，建议只在参考图片或者驱动视频中的人脸占比很小的时候，才将facecrop_ratio调整为较小的值.为1 或者0 时不裁切

**2024/11/01**    
* Add upscale model and Resnet model auto download codes（if had ，they in comfyUI/models/upscale_models/RealESRGAN_x2plus.pth and comfyUI/models/Hallo/facelib/detection_Resnet50_Final.pth）， first use ，keep “realesrgan” and “face_detection_model” ‘none’ will auto download..  
* 增加detection_Resnet50_Final.pth 和RealESRGAN_x2plus.pth自动下载的代码，首次使用，保持realesrgan和face_detection_model菜单为‘none’（无）时就会自动下载，如果菜单里已有模型，请选择模型。  

**2024/10/26**    
* 新增hallo2的2倍放大节点，输入视频的尺寸必须是512 * 512方形，输出为1024 * 1024

---

# 1. Installation

In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_EchoMimic.git
```

---
  
# 2. Requirements  

```
pip install -r requirements.txt
pip install --no-deps facenet-pytorch
```
Notice
---
* 如果安装facenet-pytorch后comfyUI奔溃，可以先卸载torch，然后再重新安装，以下版本只是示例：
* if comfyUI  broken after pip  install  facenet-pytorch ,try this below: 
```
pip uninstall torchaudio torchvision torch xformers
pip install torch torchvision torchaudio --index-url  https://download.pytorch.org/whl/cu124
pip install xformers
```
* 如果使用的是便携包版本在python_embeded目录下 打开CMD ;   
* If it is a  portable package comfyUI： open CMD in python_embeded dir   
```
python -m pip uninstall torchaudio torchvision torch xformers
python -m pip install torch torchvision torchaudio --index-url  https://download.pytorch.org/whl/cu124
python -m pip install xformers
```

* 如果ffmpeg 报错，if ffmpeg error：  
```
pip uninstall ffmpeg   
pip install ffmpeg-python  
```

* 其他库缺啥装啥。。。  
* If the module is missing, , pip install  missing module.       

## Troubleshooting errors with stable-audio-tools / other audio issues
**If using conda & python >3.12**
> Uninstall all & downgrade python
```
pip uninstall torchaudio torchvision torch xformers ffmpeg

conda uninstall python
conda install python=3.11.9

pip install --upgrade pip wheel
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
or install torch 2.4 
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
**Should have most of these packages if you install the custom nodes from git urls**
```
pip install flash-attn spandrel opencv-python diffusers jwt diffusers bitsandbytes omegaconf decord carvekit insightface easydict open_clip ffmpeg-python taming onnxruntime
```
---

# 3. Models Required 
----
如果能直连抱脸,点击就会自动下载所需模型,不需要手动下载.  
unet [link](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)  
other  [link](https://huggingface.co/BadToBest/EchoMimic/tree/main)   
vae(stabilityai/sd-vae-ft-mse)    [link](https://huggingface.co/stabilityai/sd-vae-ft-mse)     
hallo upscale [huggingface](https://huggingface.co/fudan-generative-ai/hallo2/tree/main)  # auto downlad

```
├── ComfyUI/models/ echo_mimic
|         ├── unet
|             ├── diffusion_pytorch_model.bin
|             ├── config.json
|         ├── audio_processor
|             ├── whisper_tiny.pt
|         ├── vae
|             ├── diffusion_pytorch_model.safetensors
|             ├── config.json

```
Audio-Drived Algo Inference     
```
├── ComfyUI/models/ echo_mimic
|         ├── denoising_unet.pth
|         ├── face_locator.pth
|         ├── motion_module.pth
|         ├── reference_unet.pth
```
Audio-Drived Algo Inference  acc
```
├── ComfyUI/models/ echo_mimic
|         ├── denoising_unet_acc.pth
|         ├── face_locator.pth
|         ├── motion_module_acc.pth
|         ├── reference_unet.pth
```

Using Pose-Drived Algo Inference  
```
├── ComfyUI/models/ echo_mimic
|         ├── denoising_unet_pose.pth
|         ├── face_locator_pose.pth
|         ├── motion_module_pose.pth
|         ├── reference_unet_pose.pth
```
Using Pose-Drived Algo Inference  ACC
```
├── ComfyUI/models/ echo_mimic
|         ├── denoising_unet_pose_acc.pth
|         ├── face_locator_pose.pth
|         ├── motion_module_pose_acc.pth
|         ├── reference_unet_pose.pth
```

---

# 4 Example
-----
示例的VH node ComfyUI-VideoHelperSuite node: [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

motion_sync Extract facial features directly from the video (with the option of voice synchronization), while generating a PKL model for the reference video ，The new version 
直接从从视频中提取面部特征(可以选择声音同步),同时生成参考视频的pkl模型  最新版    
 ![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/video2video.gif)

mormal Audio-Drived Algo Inference   The new version  workflow  音频驱动视频常规示例 最新  
![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/audio2video.png)

mormal Audio-Drived Algo Inference   The old version  workflow  音频驱动视频常规示例  2倍放大 1024*1024  旧版本示例  
![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/echonew.png)

pose from pkl，The old version, 基于预生成的pkl模型生成视频.  旧版      
 ![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/new.png)

---

# 5 Function Description
---
--infer_mode：音频驱动视频生成，“audio_drived” 和"audio_drived_acc"；      
--infer_mode：参考pkl模型文件视频pose生成 "pose_normal", "pose_acc"；   
    ----motion_sync：如果打开且video_file有视频文件时，生成pkl文件，并生成参考视频的视频；pkl文件在input\tensorrt_lite 目录下，再次使用需要重启comfyUI。   
    ----motion_sync：如果关闭且pose_mode不为none的时候，读取选定的pose_mode目录名的pkl文件，生成pose视频；如果pose_mode为空的时候，生成基于默认assets\test_pose_demo_pose的视频   
    ----audio_from_video：仅在motion_sync开启，且video_file有视频文件时可用，可用提取video_file的视频文件的声音，请确保该视频有声音，且为mp4格式。  
 
**特别的选项**：  
   --save_video：如果不想使用VH节点时，可以开启，默认关闭；     
   --draw_mouse：你可以试试；    
   --length：帧数，时长等于length/fps；     
   --acc模型 ，6步就可以，但是质量略有下降；   
   --lowvram :低显存用户可以开启 lowvram users can enable it  
   --内置内置图片等比例裁切。   
**特别注意的地方**：   
   --cfg数值设置为1，仅在turbo模式有效，其他会报错。    

**Infir_mode**: Audio driven video generation, "audio-d rived" and "audio-d rived_acc";   
**Infer_rode**: Refer to the PKL model file to generate "pose_normal" and "pose_acc" for the video pose;   
**Motion_Sync**: If opened and there is a video file in videoFILE, generate a pkl file and generate a reference video for the video; The pkl file is located in the input \ sensorrt_lite directory. To use it again, you need to restart ComfyUI.    
**Motion_Sync**: If turned off and pose_mode is not 'none', read the pkl file of the selected pose_mode directory name and generate a pose video; If pose_mode is empty, generate a video based on the default assets \ test_pose_demo_pose    
**Audio_from**-video: Only available when motion_stync is enabled and videoFILE has video files, it can extract the sound from videoFILE's video files. Please ensure that the video has sound and is in mp4 format.   
 
**Special options:**   
--**Save_video**: If you do not want to use VH nodes, it can be turned on and turned off by default;   
--**Draw_mause**: You can try it out;   
--**Length**: frame rate, duration equal to length/fps;   
--The ACC model only requires 6 steps, but the quality has slightly decreased;   
--Built in image proportional cropping.   
Special attention should be paid to:   
--The cfg value is set to 1, which is only valid in turbo mode, otherwise an error will be reported.   

---

**既往更新：**   
* 当你用torch 2.2.0+cuda 成功安装最新的opencv-python库后，可以卸载掉基于 2.2.0版本的torch torchvision torchaudio xformers 然后重新安装更高版本的torch torchvision torchaudio xformers，以下是卸载和安装的示例（假设安装torch2.4）：   
* 添加lowvram模式，方便6G或者8G显存用户使用，注意，开启之后会很慢，而且占用内存较大，请谨慎尝试。     
* 修改vae模型的加载方式，移至ComfyUI/models/echo_mimic/vae路径（详细见下方模型存放地址指示图），降低hf加载模型的优先级，适用于无梯子用户。     
  
**Previous updates：**   
* After successfully installing the latest OpenCV Python library using torch 2.2.0+CUDA, you can uninstall torch torch vision torch audio xformers based on version 2.2.0 and then reinstall a higher version of torch torch vision torch audio xformers. Here is an example of uninstallation and installation (installing torch 2.4):  
* Add lowvram mode for convenient use by 6G or 8G video memory users. Please note that it will be slow and consume a large amount of memory when turned on. Please try carefully  

  
---

6 Citation
------
EchoMimici
``` python  
@misc{chen2024echomimic,
  title={EchoMimic: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning},
  author={Zhiyuan Chen, Jiajiong Cao, Zhiquan Chen, Yuming Li, Chenguang Ma},
  year={2024},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
hallo2
```
@misc{cui2024hallo2,
	title={Hallo2: Long-Duration and High-Resolution Audio-driven Portrait Image Animation},
	author={Jiahao Cui and Hui Li and Yao Yao and Hao Zhu and Hanlin Shang and Kaihui Cheng and Hang Zhou and Siyu Zhu and️ Jingdong Wang},
	year={2024},
	eprint={2410.07718},
	archivePrefix={arXiv},
	primaryClass={cs.CV}
}
```



