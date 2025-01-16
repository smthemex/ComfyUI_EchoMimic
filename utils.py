# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import datetime
import logging
import os
import torch
from PIL import Image
import numpy as np
import cv2
import pickle
import gc
import shutil
from huggingface_hub import hf_hub_download
try:
    from moviepy.editor import  VideoFileClip, AudioFileClip
except:
    try:
        from moviepy import VideoFileClip, AudioFileClip
    except:
        from moviepy import *
import random
from .src.utils.mp_utils  import LMKExtractor
from .src.utils.motion_utils import motion_sync
from .src.utils.util import save_videos_grid, crop_and_pad,crop_and_pad_rectangle,center_crop
from .echomimic_v2.src.utils.dwpose_util import draw_pose_select_v2
from comfy.utils import common_upscale,ProgressBar
import folder_paths
from multiprocessing.pool import ThreadPool

weight_dtype = torch.float16
cur_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tensorrt_lite= os.path.join(folder_paths.get_input_directory(),"tensorrt_lite")
MAX_SIZE = 768

def process_video_v2(ref_image_pil, uploaded_audio, width, height, length, seed,
                   context_frames, context_overlap, cfg, steps, sample_rate, fps, pipe,
                  save_video, pose_dir, audio_file_prefix,visualizer,video_images,face_detector):
    origin_h = height
    origin_w = width
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
                logging.info(
                    " *** Start input image align . 基于基准图片，开始输入图片的对齐! ***")
                input_img_left_eye_y_ = input_img_left_eye_y[0] #基于1024的绝对值
                base_image_left_eye_y_ = base_image_left_eye_y[0]#基于1024的绝对值
               
                panding_img=align_img(base_image_length, input_img_length, 1024, 1024, panding_img_align, base_image_left_eye_y_,
                          input_img_left_eye_y_)
                
            else:  # 人体比例接近，但是高度不对，也需要对齐
                logging.info(
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
        logging.info("***** start infer video to npy files for drive pose ! ***** ")
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
                logging.info("*********first frame don't has person,skip it**********")
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
                    logging.info(
                        "Starting the first frame gesture alignment based on the input image *** 基于输入图片，开始首帧手势缩放对齐 !")
                    input_left_eye_y_ = input_left_eye_y[0]
                    first_left_eye_y_ = first_left_eye_y[0]
                    input_frames_cv2 = [align_img(input_length, first_length, height, width, i, input_left_eye_y_,
                                                  first_left_eye_y_) for i in input_frames_cv2]
                else:  # 人体比例接近，但是高度不对，也需要对齐
                    logging.info(
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
            logging.info(f"use default pose {pose_dir} for running !")
            pose_dir = os.path.join(cur_path, f"echomimic_v2/assets/halfbody_demo/pose/{pose_d}")
            USE_Default = True
        else:
            logging.info(
                "Use NPY files for custom videos, which must be located in directory 'comfyui/input/tensorrt_lite'")
            pose_dir = os.path.join(tensorrt_lite, pose_dir)
            USE_Default=False if "sapiens" in pose_dir else True
            
        
    if seed is not None and seed > -1:
        generator = torch.manual_seed(seed)
    else:
        generator = torch.manual_seed(random.randint(100, 1000000))
    
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
    
    poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
    print(f"poses_tensor:{poses_tensor.shape}")
    # audio_clip = AudioFileClip(audio_clip)
    #
    # audio_clip = audio_clip.set_duration(L / fps)
 
    video = pipe(
        infer_image_pil,
        uploaded_audio,
        poses_tensor[:, :, :L, ...],
        width,
        height,
        L,
        steps,
        cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=context_frames,
        fps=fps,
        context_overlap=context_overlap,
        start_idx=start_idx,
    ).videos
    
    final_length = min(video.shape[2],poses_tensor.shape[2], L)
    video_sig = video[:, :, :final_length, :, :]
    output_file = os.path.join(folder_paths.output_directory, f"{audio_file_prefix}_echo.mp4")
    print(f"**** final_length is : {final_length} ****")
    
    if origin_h!=origin_w:
        ouput_list = save_videos_grid(video_sig, output_file, n_rows=1, fps=fps, save_video=save_video,size=(origin_w,origin_h),ref_image_pil=ref_image_pil)
    else:
        ouput_list = save_videos_grid(video_sig, output_file, n_rows=1, fps=fps, save_video=save_video)

    if save_video:
        output_video_path = os.path.join(folder_paths.output_directory, f"{audio_file_prefix}_audio.mp4")
        video_clip = VideoFileClip(output_file)
        audio_clip = AudioFileClip(uploaded_audio)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(
            output_video_path,
            codec="libx264", audio_codec="aac")
        print(f"**** saving{output_file} at {output_video_path} ****")
        video_clip.reader.close()
        audio_clip.close()
        final_clip.reader.close()
    return ouput_list


def process_video(face_img, uploaded_audio, width, height, length, seed, facemask_dilation_ratio,
                  facecrop_dilation_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps, pipe,
                  face_detector, save_video, pose_dir, video_images, audio_file_prefix,
                  visualizer=None,):
    if seed is not None and seed > -1:
        generator = torch.manual_seed(seed)
    else:
        generator = torch.manual_seed(random.randint(100, 1000000))
    
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
        
    video = pipe(
        ref_image_pil,
        uploaded_audio,
        face_mask_tensor,
        width,
        height,
        length,
        steps,
        cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=context_frames,
        fps=fps,
        context_overlap=context_overlap
    ).videos
    
    final_length = min(video.shape[2], face_mask_tensor.shape[2], length)
    output_file = os.path.join(folder_paths.output_directory, f"{audio_file_prefix}_echo.mp4")
    print(f"**** final_length is : {final_length} ****")
    
    ouput_list = save_videos_grid(video, output_file, n_rows=1, fps=fps, save_video=save_video)
    
    if save_video:
        output_video_path = os.path.join(folder_paths.output_directory, f"{audio_file_prefix}_audio.mp4")
        video_clip = VideoFileClip(output_file)
        audio_clip = AudioFileClip(uploaded_audio)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(
            output_video_path,
            codec="libx264", audio_codec="aac")
        print(f"**** saving{output_file} at {output_video_path} ****")
        video_clip.reader.close()
        audio_clip.close()
        final_clip.reader.close()
    
    return ouput_list

def motion_sync_main(vis, width, height, video_images, face_img,facecrop_dilation_ratio, audio_file_prefix):
    lmk_extractor = LMKExtractor()
    ref_det = lmk_extractor(face_img)
    
    #driver_video = os.path.join(folder_paths.input_directory, driver_video)
    # if audio_form_video:
    #     audio_path = os.path.join(folder_paths.input_directory, f"{audio_file_prefix}_audio.wav")
    #     video_clip = VideoFileClip(driver_video)
    #     audio_clip = video_clip.audio
    #     audio_clip.write_audiofile(audio_path)
    #     video_clip.close()
    #     audio_clip.close()
    
    video_len,_,_,_=video_images.size()
    if video_len<25:
        raise "input video has not much frames for driver,change your input video!"
    else:
        tensor_list = list(torch.chunk(video_images, chunks=video_len))
        input_frames_cv2=[tensor2cv(tensor_upscale(i, width, height)) for i in tensor_list]
        
    # print(ref_det)
    sequence_driver_det = []
    if input_frames_cv2:
        try:
            print("**** Starting process video ****")
            for frame in input_frames_cv2:
                result = lmk_extractor(frame)
                assert result is not None, "bad video, face not detected"
                sequence_driver_det.append(result)
        except:
            print("face detection failed")
    else:
        raise "input video error,change your input video!"
    
    print("**** motion sync lenght " f"{len(sequence_driver_det)} ****")
  
    if vis:
        if facecrop_dilation_ratio==0:
            facecrop_dilation_ratio=1
        pose_frames_driver = [vis.draw_landmarks((width, height), i["lmks"],facecrop_dilation_ratio, normed=True) for i in sequence_driver_det]
        poses_add_driver = [(i * 0.5 + j * 0.5).clip(0, 255).astype(np.uint8) for i, j in
                            zip(input_frames_cv2, pose_frames_driver)]
        #print(f"**** poses_add_driver is done in len : {len(poses_add_driver)} ****")
        
    save_dir = os.path.join(tensorrt_lite, audio_file_prefix)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        sequence_det_ms = motion_sync(sequence_driver_det, ref_det)
        for i in range(len(sequence_det_ms)):
            with open('{}/{}.pkl'.format(save_dir, i), 'wb') as file:
                pickle.dump(sequence_det_ms[i], file)
        print(f"**** motion_sync {save_dir} is done ****")
    else:  #即便有文件夹，还是重新生成，避免出错
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir=os.path.join(tensorrt_lite,f"{audio_file_prefix}_{timestamp}")
        os.makedirs(save_dir)
        sequence_det_ms = motion_sync(sequence_driver_det, ref_det)
        for i in range(len(sequence_det_ms)):
            with open('{}/{}.pkl'.format(save_dir, i), 'wb') as file:
                pickle.dump(sequence_det_ms[i], file)
        print(f"**** motion_sync {save_dir} is done ****")
    return save_dir,video_len


def select_face(det_bboxes, probs):
    ## max face from faces that the prob is above 0.8
    ## box: xyxy
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None
    sorted_bboxes = sorted(filtered_bboxes, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]


def find_directories(base_path):
    directories = []
    for root, dirs, files in os.walk(base_path):
        for name in dirs:
            directories.append(name)
    return directories

def download_weights(file_dir,repo_id,subfolder="",pt_name=""):
    if subfolder:
        file_path = os.path.join(file_dir,subfolder, pt_name)
        sub_dir=os.path.join(file_dir,subfolder)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        if not os.path.exists(file_path):
            file_path = hf_hub_download(
                repo_id=repo_id,
                subfolder=subfolder,
                filename=pt_name,
                local_dir = file_dir,
            )
        return file_path
    else:
        file_path = os.path.join(file_dir, pt_name)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if not os.path.exists(file_path):
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=pt_name,
                local_dir=file_dir,
            )
        return file_path


def pil2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img


def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = pil2narry(value)
        list_in[i] = modified_value
    return list_in


def get_video_img(tensor):
    if tensor == None:
        return None
    outputs = []
    for x in tensor:
        x = tensor_to_pil(x)
        outputs.append(x)
    yield outputs


def gen_img_form_video(tensor):
    pil = []
    for x in tensor:
        pil[x] = tensor_to_pil(x)
    yield pil


def phi_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        list_in[i] = value
    return list_in

def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def nomarl_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_pil(samples)
    return img_pil

def tensor_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples

def tensor2cv(tensor_image):
    if len(tensor_image.shape)==4:# b hwc to hwc
        tensor_image=tensor_image.squeeze(0)
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()
    tensor_image=tensor_image.numpy()
    #反归一化
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def cvargb2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def cv2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def images_generator(img_list: list,):
    #get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_,Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_,np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    
    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in=img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in,np.ndarray):
            i=cv2.cvtColor(img_in,cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            #print(i.shape)
            return i
        else:
           raise "unsupport image list,must be pil,cv2 or tensor!!!"
        
    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image

def load_images(img_list: list,):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images

def tensor2pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image


def cf_tensor2cv(tensor,width, height):
    d1, _, _, _ = tensor.size()
    if d1 > 1:
        tensor_list = list(torch.chunk(tensor, chunks=d1))
        tensor = [tensor_list][0]
    cr_tensor=tensor_upscale(tensor,width, height)
    cv_img=tensor2cv(cr_tensor)
    return cv_img

def center_resize_pad(img, new_width, new_height):#为简化，new已是正方形
    h, w = img.shape[:2]
    if w == h:
        if w == new_width:
            return img
        else:
            return cv2.resize(img, (new_width, new_height))
    else: #蒙版也有可能不是正方形
        if h > w:  # 竖直图左右填充
            s = max(h, w)
            f = np.zeros((s, s, 3), np.uint8)
            ax, ay = (s - img.shape[1]) // 2, (s - img.shape[0]) // 2
            f[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img
        else:
            f = center_crop(img, h, h)
        return cv2.resize(f, (new_width, new_height))
       

def img_padding(height,width,ref_image_pil):
    output_max = max(height, width)
    img = tensor2cv(ref_image_pil)
    h, w = img.shape[:2]
    if h==w:
        return cv2.resize(img, (output_max,output_max), interpolation=cv2.INTER_AREA)
    else:
        if h > w: #竖直图左右填充
            s = max(h, w)
            f = np.zeros((s, s, 3), np.uint8)
            ax, ay = (s - w) // 2, (s - h) // 2
            f[ay:h + ay, ax:ax + w] = img
        else:
            f=center_crop(img, h, h)
        
        return cv2.resize(f, (output_max,output_max), interpolation=cv2.INTER_AREA)
        
def estimate_ratio(keypoint: list,box_xy,length=None):
    x1, y1, x2, y2=box_xy
    bbox_width, bbox_height = x2 - x1, y2 - y1
    left_eye_y = []
    left_shoulder_y = []
    left_elbow_y = []
    for i, (name, (x, y, conf)) in enumerate(keypoint[0].items()):
        if name == "left_eye":
            if conf > 0.3:
                y_coord =int(y * bbox_height / 256) + y1 #获取坐标点在原图的绝对值
                left_eye_y.append(y_coord)
        if name == "left_shoulder":
            if conf > 0.3:
                y_coord = int(y * bbox_height / 256) + y1
                left_shoulder_y.append(y_coord)
        if name == "left_elbow":
            if conf > 0.3:
                y_coord = int(y * bbox_height / 256) + y1
                left_elbow_y.append(y_coord)
    
    if left_eye_y and left_elbow_y:
        length=left_elbow_y[0] - left_eye_y[0]
    elif left_eye_y and left_shoulder_y and not left_elbow_y :
        length = left_shoulder_y[0] - left_eye_y[0]
    else:
        pass
    if left_eye_y:
        left_eye_y=[left_eye_y[0]]
    return length,left_eye_y


def align_img(input_length, first_length, height, width, input_frames_cv2_first, input_left_eye_y, first_left_eye_y):
    ratio = input_length / first_length  #603 [201] 679 [220]
    input_frames_cv2_first = cv2.resize(input_frames_cv2_first, (int(height * ratio), int(height * ratio)),
                                        interpolation=cv2.INTER_AREA)  # 缩小
    base_image=np.zeros((height, width,3), np.uint8)
    if input_length / height < first_length / height:  # 输入图的人物占比要小，pose图需要缩小0.88对齐,#603 [201] 679 [220]
        reduced_image,pad_size=center_paste(base_image, input_frames_cv2_first) #中心粘贴，pad为黑边尺寸
        eye_y=int(first_left_eye_y*ratio+pad_size[1]) #图片缩放后的绝对值

    else:  # pose图里人物的比例小于输入图，pose要放大
        reduced_image=center_crop(input_frames_cv2_first, height, width) #中心裁切
        h, w = input_frames_cv2_first.shape[:2]
        shift_y=(h-height)//2 #图片放大后裁切掉的边界
        eye_y= int(first_left_eye_y*ratio-shift_y)
        
    move_ = -int(eye_y - input_left_eye_y) if eye_y>= int(input_left_eye_y) else int(input_left_eye_y- eye_y) #对齐眼睛
    translation_matrix = np.float32([[1, 0, 0], [0, 1, move_]])
    shifted_image = cv2.warpAffine(reduced_image, translation_matrix, (width, height))
    
    return shifted_image

def center_paste(img_b,img_f):
    b_h,b_w=img_b.shape[:2]
    f_h, f_w = img_f.shape[:2]
    x = (b_w - f_w) // 2
    y = (b_h - f_h) // 2
    # 确保坐标不会是负数
    x = max(0, x)
    y = max(0, y)
    img_b[y:y + f_h, x:x + f_w] = img_f
    return img_b,(x,y)


def affine_img(input_left_eye_y, first_left_eye_y, img):
    height,width=img.shape[:2]
    move_ = int(input_left_eye_y[0] - first_left_eye_y[0])
    translation_matrix = np.float32([[1, 0, 0], [0, 1, move_]])  # y轴位移
    shifted_image = cv2.warpAffine(img, translation_matrix, (width, height))
    return shifted_image


def convert_fps(src_path, tgt_path, tgt_fps=24, tgt_sr=16000):
    clip = VideoFileClip(src_path)
    new_clip = clip.set_fps(tgt_fps)
    if tgt_fps is not None:
        audio = new_clip.audio
        audio = audio.set_fps(tgt_sr)
        new_clip = new_clip.set_audio(audio)
    if '.mov' in tgt_path:
        tgt_path = tgt_path.replace('.mov', '.mp4')
    new_clip.write_videofile(tgt_path, codec='libx264', audio_codec='aac')


def get_video_pose(visualizer,
        frames,
        sample_stride: int = 1,
        max_frame=None):
    # read input video
    #vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    #sample_stride *= max(1, int(vr.get_avg_fps() / 24))
    
    #frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    #frames=vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    # print(frames[0])
    if max_frame is not None:
        frames = frames[:max_frame]
    height, width, _ = frames[0].shape
    print("start get video pose list")
    detected_poses = [visualizer(frm) for frm in frames]
    visualizer.release_memory()
    
    return detected_poses, height, width, frames


def resize_and_pad(img, max_size):
    img_new = np.zeros((max_size, max_size, 3)).astype('uint8')
    imh, imw = img.shape[0], img.shape[1]
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw / imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half - half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh / imw * imw_new))
        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half - half_h
        re = rb + imh_new
    
    img_resize = cv2.resize(img, (imw_new, imh_new))
    img_new[rb:re, cb:ce, :] = img_resize
    return img_new


def resize_and_pad_param(imh, imw, max_size):
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw / imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half - half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh / imw * imw_new))
        imh_new = max_size
        
        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half - half_h
        re = rb + imh_new
    
    return imh_new, imw_new, rb, re, cb, ce


def get_pose_params(detected_poses, max_size):
    
    print('get_pose_params...')
    # pose rescale
    height = 768
    width=768
    
    w_min_all, w_max_all, h_min_all, h_max_all = [], [], [], []
    mid_all = []
    for num, detected_pose in enumerate(detected_poses):
        detected_poses[num]['num'] = num
        candidate_body = detected_pose['bodies']['candidate']
        score_body = detected_pose['bodies']['score']
        candidate_face = detected_pose['faces']
        score_face = detected_pose['faces_score']
        candidate_hand = detected_pose['hands']
        score_hand = detected_pose['hands_score']
        
        # face
        if candidate_face.shape[0] > 1:
            index = 0
            candidate_face = candidate_face[index]
            score_face = score_face[index]
            detected_poses[num]['faces'] = candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1])
            detected_poses[num]['faces_score'] = score_face.reshape(1, score_face.shape[0])
        else:
            candidate_face = candidate_face[0]
            score_face = score_face[0]
        
        # body
        if score_body.shape[0] > 1:
            tmp_score = []
            for k in range(0, score_body.shape[0]):
                tmp_score.append(score_body[k].mean())
            index = np.argmax(tmp_score)
            candidate_body = candidate_body[index * 18:(index + 1) * 18, :]
            score_body = score_body[index]
            score_hand = score_hand[(index * 2):(index * 2 + 2), :]
            candidate_hand = candidate_hand[(index * 2):(index * 2 + 2), :, :]
        else:
            score_body = score_body[0]
        all_pose = np.concatenate((candidate_body, candidate_face))
        all_score = np.concatenate((score_body, score_face))
        all_pose = all_pose[all_score > 0.8]
        
        body_pose = np.concatenate((candidate_body,))
        mid_ = body_pose[1, 0]
        
        face_pose = candidate_face
        hand_pose = candidate_hand
        
        h_min, h_max = np.min(face_pose[:, 1]), np.max(body_pose[:7, 1])
        
        h_ = h_max - h_min
        
        mid_w = mid_
        w_min = mid_w - h_ // 2
        w_max = mid_w + h_ // 2
        
        w_min_all.append(w_min)
        w_max_all.append(w_max)
        h_min_all.append(h_min)
        h_max_all.append(h_max)
        mid_all.append(mid_w)
    
    w_min = np.min(w_min_all)
    w_max = np.max(w_max_all)
    h_min = np.min(h_min_all)
    h_max = np.max(h_max_all)
    mid = np.mean(mid_all)
    
    margin_ratio = 0.25
    h_margin = (h_max - h_min) * margin_ratio
    
    h_min = max(h_min - h_margin * 0.8, 0)
    h_max = min(h_max + h_margin * 0.1, 1)
    
    h_new = h_max - h_min
    
    h_min_real = int(h_min * height)
    h_max_real = int(h_max * height)
    mid_real = int(mid * width)
    
    height_new = h_max_real - h_min_real + 1
    width_new = height_new
    w_min_real = mid_real - width_new // 2
    if w_min_real < 0:
        w_min_real = 0
        width_new = mid_real * 2
    
    w_max_real = w_min_real + width_new
    w_min = w_min_real / width
    w_max = w_max_real / width
    
    imh_new, imw_new, rb, re, cb, ce = resize_and_pad_param(height_new, width_new, max_size)
    res = {'draw_pose_params': [imh_new, imw_new, rb, re, cb, ce],
           'pose_params': [w_min, w_max, h_min, h_max],
           'video_params': [h_min_real, h_max_real, w_min_real, w_max_real],
           }
    return res


def save_pose_params_item(input_items):
    detected_pose, pose_params, draw_pose_params, save_dir = input_items
    w_min, w_max, h_min, h_max = pose_params
    num = detected_pose['num']
    candidate_body = detected_pose['bodies']['candidate']
    candidate_face = detected_pose['faces'][0]
    candidate_hand = detected_pose['hands']
    candidate_body[:, 0] = (candidate_body[:, 0] - w_min) / (w_max - w_min)
    candidate_body[:, 1] = (candidate_body[:, 1] - h_min) / (h_max - h_min)
    candidate_face[:, 0] = (candidate_face[:, 0] - w_min) / (w_max - w_min)
    candidate_face[:, 1] = (candidate_face[:, 1] - h_min) / (h_max - h_min)
    candidate_hand[:, :, 0] = (candidate_hand[:, :, 0] - w_min) / (w_max - w_min)
    candidate_hand[:, :, 1] = (candidate_hand[:, :, 1] - h_min) / (h_max - h_min)
    detected_pose['bodies']['candidate'] = candidate_body
    detected_pose['faces'] = candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1])
    detected_pose['hands'] = candidate_hand
    detected_pose['draw_pose_params'] = draw_pose_params
    np.save(save_dir + '/' + str(num) + '.npy', detected_pose)


def save_pose_params(detected_poses, pose_params, draw_pose_params, save_dir):
    #save_dir = ori_video_path.replace('video', 'pose/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    input_list = []
    
    for i, detected_pose in enumerate(detected_poses):
        input_list.append([detected_pose, pose_params, draw_pose_params, save_dir])
    
    pool = ThreadPool(8)
    pool.map(save_pose_params_item, input_list)
    pool.close()
    pool.join()
    return save_dir


def get_img_pose(visualizer,
        img_path: str,
        sample_stride: int = 1,
        max_frame=None):
    # read input img
    frame = cv2.imread(img_path)
    height, width, _ = frame.shape
    short_size = min(height, width)
    resize_ratio = max(MAX_SIZE / short_size, 1.0)
    frame = cv2.resize(frame, (int(resize_ratio * width), int(resize_ratio * height)))
    height, width, _ = frame.shape
    detected_poses = [visualizer(frame)]
    visualizer.release_memory()
    
    return detected_poses, height, width, frame


def save_aligned_img(ori_frame, video_params, max_size):
    h_min_real, h_max_real, w_min_real, w_max_real = video_params
    img = ori_frame[h_min_real:h_max_real, w_min_real:w_max_real, :]
    img_aligened = resize_and_pad(img, max_size=max_size)
    print('aligned img shape:', img_aligened.shape)
    save_dir = './assets/refimg_aligned'
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'aligned.png')
    cv2.imwrite(save_path, img_aligened)
    return save_path
