# ComfyUI_EchoMimic
You can using EchoMimic in comfyui

EchoMimicin：Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning  

EchoMimicin  From: [EchoMimic](https://github.com/BadToBest/EchoMimic/tree/main)

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

Update：
---
2024/07/17   
--- 动作同步(motion_sync)功能已上线,你可以加载想同步的视频,提取其面部特征及说话姿态等..   
---motion_syn 的生成的pkl文件,默认存储在input/tensorrt_lite文件夹下,  
--- 有几种模式可以选,比如基于视频的音画同步(开启audio_form_video,开启motion_sync,选择参考视频),仅画面同步(关闭audio_form_video,开启motion_sync,选择参考视频),pose参考其他pkl模式(关闭audio_form_video,关闭motion_sync,pose_dir选择pose目录)    
--- 支持加速模型..  
--- motion_sync is working, I am very busy, so I did not test it in detail.     
---The generated pkl file for motion_styn is stored by default in the input/sensorrt_lite folder,      
---There are several modes to choose from, such as video based audio and image synchronization (turn on audio_form-video, turn on motion_stync, select reference video), only picture synchronization (turn off audio_form-video, turn on motion_stync, select reference video), pose reference to other pkl modes (turn off audio_form_fideo, turn off motion_stync, select pose directory for pose dir)   

--- 
Pose的错误已修正，与VH节点连接的音频目前有bug，如果需要合成声音的结果，请打开save_video；     
接入comfyUI的音频加载节点；    
The pose error has been corrected, and there is currently a bug in the audio connected to the VH node. If you need to synthesize the sound results, please open save_video     
Connect to the audio loading node of ComfyUI;   


1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_EchoMimic.git
```  
  
2.requirements  
----
```
pip install -r requirements.txt

```
or 
pip  uninstall torchaudio torchvision torch xformers   
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121   
pip install xformers==0.0.24   
pip install facenet_pytorch    

diffuser >0.26 or 0.29 is best   
如果ffmpeg 报错，if ffmpeg error：  
```
pip uninstall ffmpeg   
pip install ffmpeg-python  
```
我改了官方的diffuser支持，不支持官方支持的0.24版本，最好不要用0.24版的   
这个方法的torch最高支持2.2.0，因为facenet_pytorch最高支持2.2.0，所以最要玩这个，最好是先卸载，再安装以上的python库。cu版本低的可以换成cu181     
The torch of this method supports up to 2.2.0, because facenet_pytorch supports up to 2.2.0, so it is best to uninstall it first and then install the above Python libraries. The lower version of CU can be replaced with CU181     
缺啥装啥。。。  
If the module is missing, , pip install  missing module.       

3 Need  model 
----
Audio-Drived Algo Inference   
unet [link](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)  
other  [link](https://huggingface.co/BadToBest/EchoMimic/tree/main)   
```
├── ComfyUI/models/  
|     ├──echo_mimic
|         ├── unet
|             ├── diffusion_pytorch_model.bin
|             ├── config.json
|         ├── audio_processor
|             ├── whisper_tiny.pt
|         ├── denoising_unet.pth
|         ├── face_locator.pth
|         ├── motion_module.pth
|         ├── reference_unet.pth
```
vae    
stabilityai/sd-vae-ft-mse  [link](https://huggingface.co/stabilityai/sd-vae-ft-mse) 

if using Pose-Drived Algo Inference  
```
├── ComfyUI/models/  
|     ├──echo_mimic
|         ├── unet
|             ├── diffusion_pytorch_model.bin
|             ├── config.json
|         ├── audio_processor
|             ├── whisper_tiny.pt
|         ├── denoising_unet_pose.pth
|         ├── face_locator_pose.pth
|         ├── motion_module_pose.pth
|         ├── reference_unet_pose.pth
```
if using Pose-turbo   
```
├── ComfyUI/models/  
|     ├──echo_mimic
|         ├── unet
|             ├── diffusion_pytorch_model.bin
|             ├── config.json
|         ├── audio_processor
|             ├── whisper_tiny.pt
|         ├── denoising_unet_pose_acc.pth
|         ├── face_locator_pose.pth
|         ├── motion_module_pose_acc.pth
|         ├── reference_unet_pose.pth
```


Example
-----
mormal workflow   
![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/normal.gif)

motion_sync  
 ![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/motion_sync.png)

using  motion_sync  cpl   pose   
 ![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/using%20motion%20pose.png)

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



