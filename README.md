# ComfyUI_EchoMimic
You can using EchoMimic in comfyui

EchoMimicin：Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning  

EchoMimicin  From: [EchoMimic](https://github.com/BadToBest/EchoMimic/tree/main)

---
## Updates:
**2024/09/02**    
* After successfully installing the latest OpenCV Python library using torch 2.2.0+CUDA, you can uninstall torch torch vision torch audio xformers based on version 2.2.0 and then reinstall a higher version of torch torch vision torch audio xformers. Here is an example of uninstallation and installation (installing torch 2.4):  
* 当你用torch 2.2.0+cuda 成功安装最新的opencv-python库后，可以卸载掉基于 2.2.0版本的torch torchvision torchaudio xformers 然后重新安装更高版本的torch torchvision torchaudio xformers，以下是卸载和安装的示例（假设安装torch2.4）：   
```
pip uninstall torchaudio torchvision torch xformers
pip install torch torchvision torchaudio --index-url  https://download.pytorch.org/whl/cu124
pip install  xformers
```
   
**Previous updates：**   
* Add lowvram mode for convenient use by 6G or 8G video memory users. Please note that it will be slow and consume a large amount of memory when turned on. Please try carefully  
* Add model support for audio acc, face crop support for pose, 0.24 diffuser import support. If there are import errors for other versions of diffusers, please leave an issue message, Cleared some code, waiting to add background paste function,   
* Fixed the bug where motion_stync is not enabled, and save_video is now turned off by default;     
* Fix the incorrect path definition for model download and the error in storing the pkl file path;     
* Change the audio output to the unified format of ComfyUI (which can now be directly connected to the latest version of VH)      

---
**既往更新：**   
* 添加lowvram模式，方便6G或者8G显存用户使用，注意，开启之后会很慢，而且占用内存较大，请谨慎尝试。     
* 修改vae模型的加载方式，移至ComfyUI/models/echo_mimic/vae路径（详细见下方模型存放地址指示图），降低hf加载模型的优先级，适用于无梯子用户。     
* 解决可能是batch图片输入的错误。   
* 加入audio acc 的模型支持，加入pose的face crop支持，0.24diffuser导入支持，其他版本的diffuser如果有导入出错，请issue留言。，清理了一些代码，待加入背景粘贴功能，     
* 修复motion_sync不启用的bug，save_video现在默认关闭；   
* 修复模型下载的路径定义错误，修复pkl文件路径存放的错误；     
* 将audio输出改成comfyUI的统一格式（已经可以直连最新版的VH）     

# 1. Installation
-----
In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_EchoMimic.git
```  
  
# 2. Requirements  
----
```
pip install -r requirements.txt
pip install opencv-python
```
Notice
---
opencv-python的最高支持版本是torch 2.2.0,,如果你的torch版本较高,首次安装时, 可以用--no-deps torch 忽略torch的安装 ,或者直接安装,然后删掉torch再安装高版本的torch,
如果是便携包，需要在python_embeded目录下，运行python -m pip install XXX 或者python -m pip uninstall XXX，以下是示例:

The highest supported version of OpenCV Python is Torch 2.2.0. If your Torch version is higher, during the first installation, you can use -- no deps Torch to ignore the installation of Torch, or install it directly, then delete Torch and install a higher version of Torch,
If it is a portable package, you need to run python - m pip install XXX or python - m pip uninstall XXX in the python-embedded directory. Here is an example:
torch2.4  
```
pip uninstall torchaudio torchvision torch xformers
pip install torch torchvision torchaudio --index-url  https://download.pytorch.org/whl/cu124
pip install xformers xformers==0.0.26
```
torch2.3  
```
pip uninstall torchaudio torchvision torch xformers
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.26  
```

如果安装 opencv-python 后comfyUI无法正常打开，请按以下命令先卸载，再安装：     
If comfyUI cannot be opened properly after installing OpenCV Python, please uninstall it first and then install it using the following command:       
```
pip uninstall torchaudio torchvision torch xformers      
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121   
pip install xformers==0.0.24   
```
如果是便携包， 需要在python_embeded目录下，运行python -m pip install XXX 或者python -m pip uninstall XXX，以下是示例   
if using python_embeded comfyUI,need in python_embeded open CMD ,and python -m pip install python_embeded，   
```
python -m pip uninstall torchaudio torchvision torch xformers   
python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121   
python -m pip install xformers==0.0.24   
```
or 或者  
delete python_embeded/Lib/site-packages name"torchaudio,torchvision,torch,xformers "dir     
或者直接删除python_embeded/Lib/site-packages 下面的torchaudio,torchvision,torch,xformers目录，然后按以下命令安装：     
```
python pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 --target"X:/XXX/XX/python_embeded/Lib/site-packages"
python pip install xformers==0.0.24 --target"X:/XXX/XX/python_embeded/Lib/site-packages"
```
other:

如果ffmpeg 报错，if ffmpeg error：  
```
pip uninstall ffmpeg   
pip install ffmpeg-python  
```

* After successfully installing the latest OpenCV Python library using torch 2.2.0+CUDA, you can uninstall torch torch vision torch audio xformers based on version 2.2.0 and then reinstall a higher version of torch torch vision torch audio xformers. Here is an example of uninstallation and installation (installing torch 2.4):  
* 当你用torch 2.2.0+cuda 成功安装最新的opencv-python库后，可以卸载掉基于 2.2.0版本的torch torchvision torchaudio xformers 然后重新安装更高版本的torch torchvision torchaudio xformers，以下是卸载和安装的示例（假设安装torch2.4）：   
```
pip uninstall torchaudio torchvision torch xformers
pip install torch torchvision torchaudio --index-url  https://download.pytorch.org/whl/cu124
pip install  xformers
```

其他库缺啥装啥。。。  
If the module is missing, , pip install  missing module.       

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
Audio-Drived Algo Inference     
```
├── ComfyUI/models/  
|     ├──echo_mimic
|         ├── unet
|             ├── diffusion_pytorch_model.bin
|             ├── config.json
|         ├── audio_processor
|             ├── whisper_tiny.pt
|         ├── vae
|             ├── diffusion_pytorch_model.safetensors
|             ├── config.json
|         ├── denoising_unet.pth
|         ├── face_locator.pth
|         ├── motion_module.pth
|         ├── reference_unet.pth
```
Audio-Drived Algo Inference  acc
```
├── ComfyUI/models/  
|     ├──echo_mimic
|         ├── unet
|             ├── diffusion_pytorch_model.bin
|             ├── config.json
|         ├── audio_processor
|             ├── whisper_tiny.pt
|         ├── vae
|             ├── diffusion_pytorch_model.safetensors
|             ├── config.json
|         ├── denoising_unet_acc.pth
|         ├── face_locator.pth
|         ├── motion_module_acc.pth
|         ├── reference_unet.pth
```

Using Pose-Drived Algo Inference  
```
├── ComfyUI/models/  
|     ├──echo_mimic
|         ├── unet
|             ├── diffusion_pytorch_model.bin
|             ├── config.json
|         ├── audio_processor
|             ├── whisper_tiny.pt
|         ├── vae
|             ├── diffusion_pytorch_model.safetensors
|             ├── config.json
|         ├── denoising_unet_pose.pth
|         ├── face_locator_pose.pth
|         ├── motion_module_pose.pth
|         ├── reference_unet_pose.pth
```
Using Pose-Drived Algo Inference  ACC
```
├── ComfyUI/models/  
|     ├──echo_mimic
|         ├── unet
|             ├── diffusion_pytorch_model.bin
|             ├── config.json
|         ├── audio_processor
|             ├── whisper_tiny.pt
|         ├── vae
|             ├── diffusion_pytorch_model.safetensors
|             ├── config.json
|         ├── denoising_unet_pose_acc.pth
|         ├── face_locator_pose.pth
|         ├── motion_module_pose_acc.pth
|         ├── reference_unet_pose.pth
```

Example
-----
示例的VH node ComfyUI-VideoHelperSuite node: [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

mormal Audio-Drived Algo Inference   new workflow  音频驱动视频常规示例   最新版本示例 
![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/example.png)

motion_sync Extract facial features directly from the video (with the option of voice synchronization), while generating a PKL model for the reference video ，The old version 
  
直接从从视频中提取面部特征(可以选择声音同步),同时生成参考视频的pkl模型    旧版    
 ![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/motion_sync_using_audio_from_video.png)

pose from pkl，The old version, 基于预生成的pkl模型生成视频.  旧版      
 ![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/new.png)


---

## Function Description

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

---

**Infir_mode**: Audio driven video generation, "audio-d rived" and "audio-d rived_acc";   
**Infer_rode**: Refer to the PKL model file to generate "pose_normal" and "pose_acc" for the video pose;   
**Motion_Sync**: If opened and there is a video file in videoFILE, generate a pkl file and generate a reference video for the video; The pkl file is located in the input \ sensorrt_lite directory. To use it again, you need to restart ComfyUI.    
**Motion_Sync**: If turned off and pose_mode is not 'none', read the pkl file of the selected pose_mode directory name and generate a pose video; If pose_mode is empty, generate a video based on the default assets \ test_pose_demo_pose    
**Audio_from**-video: Only available when motion_stync is enabled and videoFILE has video files, it can extract the sound from videoFILE's video files. Please ensure that the video has sound and is in mp4 format.   
 
### Special options:   
--**Save_video**: If you do not want to use VH nodes, it can be turned on and turned off by default;   
--**Draw_mause**: You can try it out;   
--**Length**: frame rate, duration equal to length/fps;   
--The ACC model only requires 6 steps, but the quality has slightly decreased;   
--Built in image proportional cropping.   
Special attention should be paid to:   
--The cfg value is set to 1, which is only valid in turbo mode, otherwise an error will be reported.   

My ComfyUI node list：
-----
1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)      
13、ComfyUI_Streamv2v_Plus node:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   
15、ComfyUI_AnyDoor node: [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
16、ComfyUI_Stable_Makeup node: [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  
17、ComfyUI_EchoMimic node:  [ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)   
18、ComfyUI_FollowYourEmoji node: [ComfyUI_FollowYourEmoji](https://github.com/smthemex/ComfyUI_FollowYourEmoji)   
19、ComfyUI_Diffree node: [ComfyUI_Diffree](https://github.com/smthemex/ComfyUI_Diffree)    
20、ComfyUI_FoleyCrafter node: [ComfyUI_FoleyCrafter](https://github.com/smthemex/ComfyUI_FoleyCrafter)   
21、ComfyUI_MooER: [ComfyUI_MooER](https://github.com/smthemex/ComfyUI_MooER)  


6 Citation
------
``` python  
@misc{chen2024echomimic,
  title={EchoMimic: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning},
  author={Zhiyuan Chen, Jiajiong Cao, Zhiquan Chen, Yuming Li, Chenguang Ma},
  year={2024},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```



