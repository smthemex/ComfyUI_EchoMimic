# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
from PIL import Image
import numpy as np
import cv2
import sys
import pickle
import torchaudio
from huggingface_hub import hf_hub_download
from moviepy.editor import VideoFileClip, AudioFileClip
import random
import logging
from .src.utils.mp_utils  import LMKExtractor
from .src.utils.img_utils import pil_to_cv2, cv2_to_pil, center_crop_cv2, pils_from_video, save_videos_from_pils, save_video_from_cv2_list
from .src.utils.motion_utils import motion_sync
from .src.utils.util import save_videos_grid, crop_and_pad
from comfy.utils import common_upscale,ProgressBar
import folder_paths

weight_dtype = torch.float16
cur_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tensorrt_lite= os.path.join(folder_paths.get_input_directory(),"tensorrt_lite")

def process_video(uploaded_img, uploaded_audio, width, height, length, seed, facemask_dilation_ratio,
                  facecrop_dilation_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps, pipe,
                  face_detector, save_video, pose_dir, video_files, audio_form_video, audio_file_prefix,
                  visualizer=None, crop_face=True, ):
    if seed is not None and seed > -1:
        generator = torch.manual_seed(seed)
    else:
        generator = torch.manual_seed(random.randint(100, 1000000))
    
    #### face musk prepare
    face_img = np.array(uploaded_img)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')
    det_bboxes, probs = face_detector.detect(face_img)
    select_bbox = select_face(det_bboxes, probs)
    if select_bbox is None:
        face_mask[:, :] = 255
    else:
        xyxy = select_bbox[:4].astype(float)  # 原方法的np版本 无法实现整形
        xyxy = np.round(xyxy).astype("int")
        rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
        r_pad = int((re - rb) * facemask_dilation_ratio)
        c_pad = int((ce - cb) * facemask_dilation_ratio)
        face_mask[rb - r_pad: re + r_pad, cb - c_pad: ce + c_pad] = 255
        
        #### face crop
        r_pad_crop = int((re - rb) * facecrop_dilation_ratio)
        c_pad_crop = int((ce - cb) * facecrop_dilation_ratio)
        crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]),
                     min(re + r_pad_crop, face_img.shape[0])]
        face_img, ori_face_rect_a = crop_and_pad(face_img, crop_rect)
        face_mask, ori_mask_rect_b = crop_and_pad(face_mask, crop_rect)  # ori_face_rect_a,ori_mask_rect_b no use
        face_img = cv2.resize(face_img, (width, height))
        face_mask = cv2.resize(face_mask, (width, height))
    
    ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
    face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device="cuda").unsqueeze(0).unsqueeze(
        0).unsqueeze(0) / 255.0
    audio = None
    if visualizer:
        # add face crop
        if crop_face:
            face_img = np.array(uploaded_img)
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            det_bboxes, probs = face_detector.detect(face_img)
            select_bbox = select_face(det_bboxes, probs)
            if select_bbox is not None:
                xyxy = select_bbox[:4].astype(float)  # 原方法的np版本 无法实现整形
                xyxy = np.round(xyxy).astype('int')
                rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
                r_pad_crop = int((re - rb) * facecrop_dilation_ratio)
                c_pad_crop = int((ce - cb) * facecrop_dilation_ratio)
                crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop),
                             min(ce + c_pad_crop, face_img.shape[1]),
                             min(re + c_pad_crop, face_img.shape[0])]
                print(crop_rect)
                face_img, ori_face_rect = crop_and_pad(face_img, crop_rect)
                face_mask, ori_face_mask_rect = crop_and_pad(face_mask, crop_rect)
                print(ori_face_rect)
                ori_face_size = (ori_face_rect[2] - ori_face_rect[0], ori_face_rect[3] - ori_face_rect[1])
                face_img = cv2.resize(face_img, (width, height))
                face_mask = cv2.resize(face_mask, (width, height))
            else:
                face_mask[:, :] = 255
        ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
        
        if pose_dir == "none":  # motion sync
            if video_files != "none":
                pose_dir, audio_from_v = motion_sync_main(visualizer, width, height, video_files, face_img,
                                                          audio_form_video, audio_file_prefix)
                if audio_form_video:
                    uploaded_audio = audio_from_v
                    waveform, sample_rate = torchaudio.load(uploaded_audio)
                    audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            else:
                pose_dir = os.path.join(cur_path, "assets", "test_pose_demo_pose")  # default
        else:
            pose_dir = os.path.join(tensorrt_lite, pose_dir)
        
        pose_list = []
        for index in range(len(os.listdir(pose_dir))):
            tgt_musk_path = os.path.join(pose_dir, f"{index}.pkl")
            with open(tgt_musk_path, "rb") as f:
                tgt_kpts = pickle.load(f)
            tgt_musk = visualizer.draw_landmarks((width, height), tgt_kpts)
            tgt_musk_pil = Image.fromarray(np.array(tgt_musk).astype(np.uint8)).convert('RGB')
            pose_list.append(
                torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device="cuda").permute(2, 0, 1) / 255.0)
        face_mask_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
    
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
    
    ouput_list = save_videos_grid(video, output_file, n_rows=1, fps=fps, save_video=save_video)
    
    if save_video:
        output_video_path = os.path.join(folder_paths.output_directory, f"{audio_file_prefix}_audio.mp4")
        video_clip = VideoFileClip(output_file)
        audio_clip = AudioFileClip(uploaded_audio)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(
            output_video_path,
            codec="libx264", audio_codec="aac")
        print(f"saving {output_video_path}")
        video_clip.reader.close()
        audio_clip.close()
        final_clip.reader.close()
    
    return ouput_list, audio

def motion_sync_main(vis, width, height, driver_video, face_img, audio_form_video, audio_file_prefix):
    lmk_extractor = LMKExtractor()
    ref_det = lmk_extractor(face_img)
    audio_path = None
    driver_video = os.path.join(folder_paths.input_directory, driver_video)
    
    if audio_form_video:
        audio_path = os.path.join(folder_paths.input_directory, f"{audio_file_prefix}_audio.wav")
        video_clip = VideoFileClip(driver_video)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)
        video_clip.close()
        audio_clip.close()
    
    input_frames_cv2 = [i for i in
                        pils_from_video(driver_video, width, height)]  # 原方法 先cv，中心剪裁转pil，再转cv再中心剪裁，然后再pil转CV很奇怪
    
    # print(ref_det)
    sequence_driver_det = []
    try:
        for frame in input_frames_cv2:
            result = lmk_extractor(frame)
            assert result is not None, "{}, bad video, face not detected".format(driver_video)
            sequence_driver_det.append(result)
    except:
        print("face detection failed")
    
    print("motion sync lenght " f"{len(sequence_driver_det)}")
    
    if vis:
        pose_frames_driver = [vis.draw_landmarks((width, height), i["lmks"], normed=True) for i in sequence_driver_det]
        poses_add_driver = [(i * 0.5 + j * 0.5).clip(0, 255).astype(np.uint8) for i, j in
                            zip(input_frames_cv2, pose_frames_driver)]
    
    save_dir = os.path.join(tensorrt_lite, audio_file_prefix)
    save_dir = os.path.normpath(save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        sequence_det_ms = motion_sync(sequence_driver_det, ref_det)
        for i in range(len(sequence_det_ms)):
            with open('{}/{}.pkl'.format(save_dir, i), 'wb') as file:
                pickle.dump(sequence_det_ms[i], file)
        print(f"motion_sync {save_dir} is done")
    else:  # 同名文件
        files_ex = os.path.join(save_dir, "0.pkl")
        files_ex = os.path.normpath(files_ex)
        if not os.path.isfile(files_ex):  # 判断是否有模型文件
            sequence_det_ms = motion_sync(sequence_driver_det, ref_det)
            for i in range(len(sequence_det_ms)):
                with open('{}/{}.pkl'.format(save_dir, i), 'wb') as file:
                    pickle.dump(sequence_det_ms[i], file)
            print(f"motion_sync {save_dir} is done")
        else:
            print("The model file already exists ")
    
    # if vis:
    #     pose_frames = [vis.draw_landmarks((width, height), i, normed=False) for i in sequence_det_ms]
    #     poses_add = [(i * 0.5 + ref_frame * 0.5).clip(0, 255).astype(np.uint8) for i in pose_frames]
    #
    # poses_cat = [np.concatenate([i, j], axis=1) for i, j in zip(poses_add_driver, poses_add)]
    # output_video_path = os.path.join(folder_paths.output_directory, f"{image_name}_.mp4")
    # if save_video:
    #     save_video_from_cv2_list(poses_cat, output_video_path, fps=fps)
    return save_dir, audio_path


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
            pt_path = hf_hub_download(
                repo_id=repo_id,
                subfolder=subfolder,
                filename=pt_name,
                local_dir = file_dir,
            )
        else:
            pt_path=get_instance_path(file_path)
        return pt_path
    else:
        file_path = os.path.join(file_dir, pt_name)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if not os.path.exists(file_path):
            pt_path = hf_hub_download(
                repo_id=repo_id,
                filename=pt_name,
                local_dir=file_dir,
            )
        else:
            pt_path=get_instance_path(file_path)
        return pt_path


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

def instance_path(path, repo):
    if repo == "":
        if path == "none":
            repo = "none"
        else:
            model_path = get_local_path(folder_paths.base_path, path)
            repo = get_instance_path(model_path)
    return repo


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

def get_local_path(comfy_file_path, model_path):
    path = os.path.join(comfy_file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform == 'win32':
        model_path = model_path.replace('\\', "/")
    return model_path

def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


def tensor2cv(tensor_image):
    if len(tensor_image.shape)==4:#bhwc to hwc
        tensor_image=tensor_image.squeeze(0)
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu().detach()
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

