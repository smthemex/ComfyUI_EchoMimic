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
2024/07/21   
--修复模型下载的路径定义错误，修复pkl文件路径存放的错误；  
--将audio输出改成comfyUI的统一格式（已经可以直连最新版的VH）  
--Fix the incorrect path definition for model download and the error in storing the pkl file path;  
--Change the audio output to the unified format of ComfyUI (which can now be directly connected to the latest version of VH)   

Function Description
--
功能1：音频驱动视频生成，将pose_mode设置成无（none）时启用；   
功能2：参考视频同步生成，同时生产pkl文件。需要将pose_mode设置成normal或turbo，开启motion_sync，在video_file选择你要同步的视频，设置pose_dir 为无（none）；   
    --tips： 如果使用参考视频自带的音频，可以开始audio_from_video，此时audio输入的音频失效，此功能仅在motion_sync有效。   
功能3：参考pkl模型文件，音频驱动视频，需要将pose_mode设置成normal或turbo,在pose_dir选择你要参考的pkl模型目录；   
    --tips： 如果pose_dir为无（none），则使用内置的pkl模型，也就是插件目录的assets\test_pose_demo_pose；   
特别的选项：  
   --save_video：如果不想使用VH节点时，可以开启，默认关闭；     
   --draw_mouse：你可以试试；    
   --length：帧数，时长等于length/fps；     
   --normal和turbo：turbo，6步可以，但是质量略有下降；   
   --内置内置图片等比例裁切。   
特别注意的地方：   
   --cfg数值设置为1，仅在turbo模式有效，其他会报错。    

Function 1: Audio driven video generation, enabled when pose mode is set to none；   
Function 2: Generate reference videos synchronously and produce PKL files at the same time. You need to set pose_mode to normal or turbo, enable motion_stync, select the video you want to synchronize in videoFILE, and set pose_dir to none；   
    --Tip: If you use the audio provided in the reference video, you can start audio_from-video. At this point, the audio input will become invalid, and this function is only effective in motion_stync.
Function 3: Refer to the pkl model file, drive the audio video, and set pose_mode to normal or turbo. Select the pkl model directory you want to refer to in pose_dir;  
    --Tip: If pose_dir is none, use the built-in pkl model, which is assets \ test_pose_demo_pose in the plugin directory；   
Special options:   
    --Save_video: If you do not want to use VH nodes, you can turn it on, and it is turned off by default；  
    --Draw_mouse: You can try it out；  
    --Length: frame rate, duration equal to length/fps；  
    --Normal and Turbo: Turbo, 6 steps are fine, but the quality has slightly decreased；  
    --Built in image proportional cropping.   
Special attention should be paid to:   
    --The cfg value is set to 1, which is only valid in turbo mode, otherwise an error will be reported. 


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
![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/base.png)

motion_sync Extract facial features directly from the video (with the option of voice synchronization), while generating a PKL model for the reference video 直接从从视频中提取面部特征(可以选择声音同步),同时生成参考视频的pkl模型  
 ![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/motion_sync_using_audio_from_video.png)

pose from pkl, 基于预生成的pkl模型生成视频.  
 ![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/normal.png)



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



