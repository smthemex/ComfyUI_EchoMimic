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
2024/07/19
zero_locator_tensor 这个形参好像代码不全，导致错误，先去掉。
修改部分需求文件内容，

2024/07/17   
--- 动作同步(motion_sync)功能已上线,你可以加载想同步的视频,提取其面部特征及说话姿态等.首次使用会生成生成的pkl文件,以及基于该视频的pose动画,pkl文件存储在input/tensorrt_lite文件夹下(重启comfyUI后调用,在pose_dir菜单中选择),       
--- audio_form_video 菜单的主要功能是提取参考视频的音频,开启时,音频节点输入的音频无效,所以使用此功能时,请确认你的参考视频有声音.   
--- pose_mode选择turbo,开启加速模型(denoising_unet_pose_acc.pth  motion_module_pose_acc.pth ),cfg 1,step 6或者更高,约50s出图.  
--- pose_dir菜单 在pose为normal或turbo时有效,模型路径指向input/tensorrt_lite,如果选择none,则是用官方默认的pose.   

---The action synchronization (motion_sync) function has been launched. You can load the video you want to synchronize, extract its facial features and speech posture. The first use will generate the generated pkl file, as well as the post animation based on the video. The pkl file is stored in the input/tensor_lite folder (called after restarting the comfyUI, select from the pose_dir menu),    
---The main function of the audio_form_fideo menu is to extract the audio of the reference video. When turned on, the audio input from the audio node is invalid. Therefore, when using this function, please confirm that your reference video has sound    
---Select turbo for pose_mode, activate the acceleration model (denoising_unet_pose acc. pth motif module pose acc. pth), cfg 1, step 6 or higher, and the image will be generated in approximately 50 seconds   
---The pose_dir menu is valid when pose is normal or turbo, and the model path points to input/sensorrt_lite. If none is selected, the official default pose will be used    


--- 
audio输出连接VH节点时有bug，如果需要合成声音的结果，请打开save_video；     
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
如果能直连抱脸,点击就会自动下载所需模型,不需要需下载.  

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

Using Pose-Drived Algo Inference  
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
Using Pose-turbo   
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
mormal Audio-Drived Algo Inference  workflow  音频驱动视频常规示例    
![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/audio.gif)

motion_sync Extract facial features directly from the video (with the option of voice synchronization), while generating a PKL model for the reference video 直接从从视频中提取面部特征(可以选择声音同步),同时生成参考视频的pkl模型  
 ![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/motion_sync_from_video.png)

pose from pkl, 基于预生成的pkl模型生成视频.  
 ![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/pose_form_pkl.png)



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



