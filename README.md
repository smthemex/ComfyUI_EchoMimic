# ComfyUI_EchoMimic
You can use EchoMimic & EchoMimic V2 in comfyui

[Echomimic](https://github.com/antgroup/echomimic/tree/main)：Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning  
[Echomimic_v2](https://github.com/antgroup/echomimic_v2): Towards Striking, Simplified, and Semi-Body Human Animation

---

## New Updates 2025-01-04:
* Hallo2的放大功能并不是必须项目，已移植其他分支，主分支已删除掉Hallo2的内容  
* The amplification function of Hallo2 is not a mandatory item, it has been ported to other branches, and the content of Hallo2 has been deleted from the main branch  

* 支持新版的ACC模型，在infer_mode里选择pose_acc开启，如果外网通畅会自动下载，你也可以从[这里](https://huggingface.co/BadToBest/EchoMimicV2/tree/main)预下载（denoising_unet_acc.pth和motion_module_acc.pth），并放在ComfyUI\models\echo_mimic\v2里，推荐的步数为6步,cfg为1，尺寸为768*768。ACC模型较大，小显存耗时可能会比较长；  
* Support the new version of ACC model, select 'pose_acc' to enable in 'infer_mode', and if the network is smooth, it will automatically download. You can also pre download from [here](https://huggingface.co/BadToBest/EchoMimicV2/tree/main) and put it in A. The recommended 'steps' are '6' ,'cfg' is '1' and the size is 768 * 768. The ACC model is relatively large, and low video memory consumption may be longer


**Previous updates：**  
* 新增输入图片跟基准图片对齐功能（选择pose_normal_sapiens时自动开启，3种驱动方式都能使用，见下面的示例图），修复之前的蒙版对齐错误。
* Added the function of aligning the input image with the reference image (automatically turned on when selecting pose_normal_sapiens, all three driving methods can be used，See the example diagram below), fixed the previous mask alignment error.

* V2版现在跟V1一样，有三种pose驱动方式，第一种，infer_mode选择audio_drive,pose_dir 选择列表里的几个默认pose，则使用默认的npy pose文件，第二种，infer_mode选择audio_drive,pose_dir 选择已有的npy文件夹（位于...ComfyUI/input/tensorrt_lite目录下），第三种，infer_mode选择pose_normal_dwpose 或pose_normal_sapiens,video_images连接视频入口，确认...ComfyUI/models/echo_mimic 下有yolov8m.pt 和sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2 模型 （见图示和example里的工作流,下载地址见后附）；
* 因为调用了sapiens的pose方法，所以需要安装yolo的库ultralytics ，安装方法：  pip install ultralytics  
* The V2 version now has three different pose driving methods, just like the V1 version. The first method is to select audio_drive for infer_mode and default poses from the list for pose_dir, using the default npy pose file. The second method is to select audio-drive for infer_mode and an existing npy folder (located in the... ComfyUI/input/tensorrt_lite directory) for pose_dir. The third method is to select 'pose_normal_dwpose' or 'pose_normal_sapiens' for infer_mode, connect to the video portal with video_images, and confirm Under ComfyUI/models/echo_mimic, there are 'YOLOV8m.pt' and 'sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2' models (see the workflow in the diagram and example,Please see the download link below)
* Because the pose method of ‘Sapiens’ was called, it is necessary to install YOLO's library ultralytics. Installation method： pip install ultralytics  
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
```

# Notice
---
If use v1 version 如果要使用V1版本： 
```
pip install --no-deps facenet-pytorch 

```
* 因为V1版本才需求facenet-pytorch，所以不使用V1版是不需要安装facenet-pytorch的，如果安装facenet-pytorch后comfyUI崩了，可以先卸载torch，然后再重新安装，以下版本只是示例：
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
**3.1 V1 & V2 Shared model v1 和 v2 共用的模型**:   
如果能直连抱脸,点击就会自动下载所需模型,不需要手动下载.  
* unet [link](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)    
* V1 & V2 audio  [link](https://huggingface.co/BadToBest/EchoMimic/tree/main)    
* vae(stabilityai/sd-vae-ft-mse)    [link](https://huggingface.co/stabilityai/sd-vae-ft-mse)          

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

**3.2 V1 models V1使用以下模型**:   
* V1 address   [link](https://huggingface.co/BadToBest/EchoMimic/tree/main)    
* Audio-Drived Algo Inference 音频驱动        
```
├── ComfyUI/models/echo_mimic
|         ├── denoising_unet.pth
|         ├── face_locator.pth
|         ├── motion_module.pth
|         ├── reference_unet.pth
Audio-Drived Algo Inference  acc  音频驱动加速版
|         ├── denoising_unet_acc.pth
|         ├── motion_module_acc.pth
```

* Using Pose-Drived Algo Inference  姿态驱动   
```
├── ComfyUI/models/echo_mimic
|         ├── denoising_unet_pose.pth
|         ├── face_locator_pose.pth
|         ├── motion_module_pose.pth
|         ├── reference_unet_pose.pth
Using Pose-Drived Algo Inference  ACC   姿态驱动加速版
|         ├── denoising_unet_pose_acc.pth
|         ├── motion_module_pose_acc.pth
```

**3.2 v2 version**   
use model below V2, Automatic download, you can manually add it 使用以下模型,使用及自动下载,你可以手动添加:    
模型地址address:[huggingface](https://huggingface.co/BadToBest/EchoMimicV2/tree/main)
```
├── ComfyUI/models/echo_mimic/v2
|         ├── denoising_unet.pth
|         ├── motion_module.pth
|         ├── pose_encoder.pth
|         ├── reference_unet.pth
if use acc 姿态驱动加速版   
|         ├── denoising_unet_acc.pth
|         ├── motion_module_acc.pth
```
YOLOm8 [download link](https://huggingface.co/Ultralytics/YOLOv8/tree/main)   
sapiens pose [download link](https://huggingface.co/facebook/sapiens-pose-1b-torchscript/tree/main)  
sapiens的pose 模型可以量化为fp16的，详细见我的sapiens插件 [地址](https://github.com/smthemex/ComfyUI_Sapiens)   
```
├── ComfyUI/models/echo_mimic
|         ├── yolov8m.pt
|         ├── sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2  or/或者 sapiens_1b_goliath_best_goliath_AP_639_torchscript_fp16.pt2
```

---

# 4 Example
-----
* 自动对齐输入图片Automatically align input images；  
![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/alignA.png)
![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/align.png)

* V2加载自定义视频驱动视频，V2 loads custom video driver videos  
![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/example.png)

* V2选择自定义pose驱动视频，V2 Choose Custom Pose Driver Video   
![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/cropC.png)

* Echomimic_v2 use default pose  new version 使用官方默认的pose文件 
![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/v2.gif)

* motion_sync Extract facial features directly from the video (with the option of voice synchronization), while generating a PKL model for the reference video ，The old version 
直接从从视频中提取面部特征(可以选择声音同步),同时生成参考视频的pkl模型  旧版   
 ![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/video2video.gif)

* mormal Audio-Drived Algo Inference   The old  version  workflow  音频驱动视频常规示例 旧版  
![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/audio2video.png)

* pose from pkl，The old  version, 基于预生成的pkl模型生成视频.  旧版      
 ![](https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example/new.png)

* 示例的 VH node ComfyUI-VideoHelperSuite node: [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

---

# 5 Function Description
---
* infer_mode：音频驱动视频生成，“audio_drived” 和"audio_drived_acc"；      
* infer_mode：参考pkl模型文件视频pose生成 "pose_normal", "pose_acc"；   
    ----motion_sync：如果打开且video_file有视频文件时，生成pkl文件，并生成参考视频的视频；pkl文件在input\tensorrt_lite 目录下，再次使用需要重启comfyUI。   
    ----motion_sync：如果关闭且pose_mode不为none的时候，读取选定的pose_mode目录名的pkl文件，生成pose视频；如果pose_mode为空的时候，生成基于默认assets\test_pose_demo_pose的视频   
 
**特别的选项**：  
  * save_video：如果不想使用VH节点时，可以开启，默认关闭；     
  * draw_mouse：你可以试试；    
  * length：帧数，时长等于length/fps；     
  * acc模型 ，6步就可以，但是质量略有下降；   
  * lowvram :低显存用户可以开启 lowvram users can enable it  
  * 内置内置图片等比例裁切。   
**特别注意的地方**：   
  * cfg数值设置为1，仅在turbo模式有效，其他会报错。    

**Infir_mode**: Audio driven video generation, "audio-d rived" and "audio-d rived_acc";   
**Infer_rode**: Refer to the PKL model file to generate "pose_normal" and "pose_acc" for the video pose;   
**Motion_Sync**: If opened and there is a video file in videoFILE, generate a pkl file and generate a reference video for the video; The pkl file is located in the input \ sensorrt_lite directory. To use it again, you need to restart ComfyUI.    
**Motion_Sync**: If turned off and pose_mode is not 'none', read the pkl file of the selected pose_mode directory name and generate a pose video; If pose_mode is empty, generate a video based on the default assets \ test_pose_demo_pose    

 
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

* 增加detection_Resnet50_Final.pth 和RealESRGAN_x2plus.pth自动下载的代码，首次使用，保持realesrgan和face_detection_model菜单为‘none’（无）时就会自动下载，如果菜单里已有模型，请选择模型。    
* 当你用torch 2.2.0+cuda 成功安装最新的facenet-pytorch库后，可以卸载掉基于 2.2.0版本的torch torchvision torchaudio xformers 然后重新安装更高版本的torch torchvision torchaudio xformers，以下是卸载和安装的示例（假设安装torch2.4）：
* facecrop_ratio的放大系数为1/facecrop_ratio，如果设置为0.5，面部会得到2倍的放大，建议只在参考图片或者驱动视频中的人脸占比很小的时候，才将facecrop_ratio调整为较小的值.为1 或者0 时不裁切  
* 添加lowvram模式，方便6G或者8G显存用户使用，注意，开启之后会很慢，而且占用内存较大，请谨慎尝试。      


**Previous updates：**   
* ﻿The magnification factor of 'facecrop-ratio' is '1/facecrop-ratio'. If set to 0.5, the face will be magnified twice. It is recommended to adjust facecrop-ratio to a smaller value only when the proportion of faces in the reference image or driving video is very small,Do not cut when it is 1 or 0;     
* After successfully installing the latest ‘facenet-pytorch’ library using torch 2.2.0+CUDA, you can uninstall torch torch vision torch audio xformers based on version 2.2.0 and then reinstall a higher version of torch、 torch vision、 torch audio xformers. Here is an example of uninstallation and installation (installing torch 2.4):  
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

EchoMimici-V2
``` python  
@misc{meng2024echomimic,
  title={EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation},
  author={Rang Meng, Xingyu Zhang, Yuming Li, Chenguang Ma},
  year={2024},
  eprint={2411.10061},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
sapiens
```
@article{khirodkar2024sapiens,
  title={Sapiens: Foundation for Human Vision Models},
  author={Khirodkar, Rawal and Bagautdinov, Timur and Martinez, Julieta and Zhaoen, Su and James, Austin and Selednik, Peter and Anderson, Stuart and Saito, Shunsuke},
  journal={arXiv preprint arXiv:2408.12569},
  year={2024}
}
```

