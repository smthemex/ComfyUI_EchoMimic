# ComfyUI_EchoMimic
You can use EchoMimic & EchoMimic V2  & EchoMimic V3 in comfyui.   
[Echomimic](https://github.com/antgroup/echomimic/tree/main)：Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning       
[Echomimic_v2](https://github.com/antgroup/echomimic_v2): Towards Striking, Simplified, and Semi-Body Human Animation   
[Echomimic_v3](https://github.com/antgroup/echomimic_v3)：1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation   


---

# New Updates 
* you can set lowram 'False ' to close mmgp 's fp8 quantum ，will get more quality output./设置lowram为false时，关闭mmgp的FP8 量化以得到更好的质量。
* add LCM support ,if set step=4（and lightX2V lora）,will run in LCM/ 步数设置为4时，自动开启LCM，当然也要lora
* v3版本新增lightX2V Lora的支持， step可以设置为10步（使用Lora时自动开启Unip）/you can use lightX2V Lora when use V3 version, set step=10; 
* 修复bug，retina-face 模型改成本地运行
* V3正式上线，测试环境12G VRAM，OOM需要减少视频分块(partial_video_length)的数值，12G可以跑65，16可以试试97，更高可以试试113
* V3 is Done,you can try it now.. need 8G and more (use mmgp,LOW LOW,partial_video_length==65 or 33)

# 1. Installation

In the ./ComfyUI /custom_nodes directory, run the following:   
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
If use v3 version 如果要使用V3版本： 
```
pip install retina-face==0.0.17 #使用须外网下载模型，待处理
pip install mmgp # optional 可选 
pip install tensorflow==2.15.0   #高版本可能会报错，存疑   
```

* 如果ffmpeg 报错，if ffmpeg error：  
```
pip uninstall ffmpeg   
pip install ffmpeg-python  
```

* 其他库缺啥装啥。。。  
* If the module is missing, , pip install  missing module.       


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
├── ComfyUI/models/vae
|             ├── diffusion_pytorch_model.safetensors or rename sd-vae-ft-mse.safetensors
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

**3.3 v3 version**   
3.3.1 from [Wan2.1-Fun-V1.1-1.3B-InP](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP/tree/main)downlaod Wan2.1_VAE.pth and diffusion_pytorch_model.safetensors   
3.3.2 use comfyui ,[clipvison-h](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/clip_vision) and [umt5_xxl_fp8_e4m3fn_scaled.safetensors ](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders)  
3.3.3 [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h/tree/main)   
3.3.4 [BadToBest/EchoMimicV3](https://huggingface.co/BadToBest/EchoMimicV3/tree/main) transformer    
3.3.5 [retinaface.h5](https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5)  目录下没有一般会自动下载
3.3.6 可选/optional lora  [kijai](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Lightx2v)
```
├── ComfyUI/models/echo_mimic/transformer 
|         ├── diffusion_pytorch_model.safetensors  # Wan2.1-Fun-V1.1-1.3B-InP transformer #3.13G 务必注意模型同名。
|         ├── config.json
├── ComfyUI/models/echo_mimic/wav2vec2-base-960h
|         ├── all config json files 
|         ├──  model.safetensors
├── ComfyUI/models/clip
|         ├── umt5_xxl_fp8_e4m3fn_scaled.safetensors
├── ComfyUI/models/clip_vision
|         ├──clipvison-h # 1.26G
├── ComfyUI/models/echo_mimic/
|         ├──diffusion_pytorch_model.safetensors  # BadToBest/EchoMimicV3
├── ComfyUI/models/vae
|         ├── Wan2.1_VAE.pth
├── ComfyUI/models/echo_mimic/.deepface/weights/    #注意.deepface前面有个点，这个是方便不能翻墙玩家
|         ├──retinaface.h5
├── ComfyUI/models/loras/    
|         ├──lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors  #KJ

```


# 4 Example
-----
* V3 version
<img src="https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example_workflows/example_v3.png" width="60%">

* V2 version

* V2加载自定义视频驱动视频，V2 loads custom video driver videos
<img src="https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example_workflows/example_v2_pose.png" width="60%">

* Echomimic_v2 use default pose  new version 使用官方默认的pose文件
<img src="https://github.com/smthemex/ComfyUI_EchoMimic/blob/main/example_workflows/example_v2_pose.png" width="60%">

* V1 version

* audio driver 音频驱动
<img src="https://github.com/smthemex/ComfyUI_EchoMimic/blob/main//example_workflows/example_v1.png" width="60%">    


* 示例的 VH node : [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

---

# 5 Function Description
---
* v1 版本的生成模式    
  a. 单纯音频驱动视频生成模式，infer_mode可选常规的“audio_drived” 和加速版"audio_drived_acc" 模型；   
  b. pose驱动生成模式，常规选项为pose_normal_sapiens/pose_normal_dwpose（等同） 加速版本为"pose_acc"模型；   
    ----motion_sync：pose驱动时，如果打开且video_file有视频文件时，生成pkl文件，并生成参考视频的视频；pkl文件在input\tensorrt_lite 目录下，再次使用需要重启comfyUI。      
    ----motion_sync：如果关闭且pose_dir不为none的时候，读取选定的pose_dir目录名的pkl文件，生成pose视频；如果pose_dir为空的时候，生成基于默认assets\test_pose_demo_pose的视频     
  
* v2 版本的生成模式   
  a. infer_mode选择audio_drive,pose_dir 选择列表里的几个默认pose，则使用默认的npy pose文件;     
  b. infer_mode选择audio_drive,pose_dir 选择已有的npy文件夹（位于...ComfyUI/input/tensorrt_lite目录下);   
  c. infer_mode选择pose_normal_dwpose 或pose_normal_sapiens,video_images连接视频入口，确认...ComfyUI/models/echo_mimic 下有yolov8m.pt 和sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2 模型,根据输入视频生成npy文件（可以下次用）和视频   

* v3 版本生成模式       
   a. 基于retina-face库生成   
   b. 如果retina-face调用失败，则以默认的女性face作为mask    

**特别的选项**：  
  * save_video：如果不想使用VH节点时，可以开启，默认关闭；     
  * draw_mouse：你可以试试；    
  * length：帧数，时长等于length/fps；     
  * acc模型 ，6步就可以，但是质量略有下降；   
  * lowvram :低显存用户可以开启 lowvram users can enable it  
  * 内置内置图片等比例裁切。
  * facecrop_ratio的放大系数为1/facecrop_ratio，如果设置为0.5，面部会得到2倍的放大，建议只在参考图片或者驱动视频中的人脸占比很小的时候，才将facecrop_ratio调整为较小的值.为1 或者0 时不裁切   
  * cfg数值设置为1，仅在turbo模式有效，其他会报错。V2推荐2.5 V3推荐3.5
  * use_mmgp 仅V3版本有效   
  * partial_video_length 仅V3版本有效，数值越低显存占用越低；
  * teacache 仅V3版本有效；     


  
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
EchoMimici-V3
```
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```
LightX2V
```
@misc{lightx2v,
 author = {LightX2V Contributors},
 title = {LightX2V: Light Video Generation Inference Framework},
 year = {2025},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/ModelTC/lightx2v}},
}
````

sapiens
```
@article{khirodkar2024sapiens,
  title={Sapiens: Foundation for Human Vision Models},
  author={Khirodkar, Rawal and Bagautdinov, Timur and Martinez, Julieta and Zhaoen, Su and James, Austin and Selednik, Peter and Anderson, Stuart and Saito, Shunsuke},
  journal={arXiv preprint arXiv:2408.12569},
  year={2024}
}
```

