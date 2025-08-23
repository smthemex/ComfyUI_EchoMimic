import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from comfy.utils import common_upscale

def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8, audio_path=None):
    import av

    save_fmt = Path(path).suffix
    #os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=int(fps))

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")



def save_videos_grid(videos: torch.Tensor, path: str, audio_path=None, rescale=False, n_rows=6, fps=8,save_video=False,size=None,ref_image_pil=None):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        z = (x * 255).numpy().astype(np.uint8) #方形
        if size:
            origin_w,origin_h=size
            if origin_h>origin_w:#竖
               z=center_crop(z,origin_w,origin_h) #竖图直接裁切
            else: # 横 max是w
                img = tensor2cv(ref_image_pil)
                h, w = img.shape[:2]
                if h>=w: #防止输入竖图或正方，想要横图的人才
                    img_bg = np.zeros((origin_h,origin_w, 3), np.uint8)
                    img_fg = cv2.resize(z, (origin_h, origin_h))  # 缩放到输出高度
                    z = img_coty2_img(img_bg, img_fg)  # 回帖
                else: #输入横图，输出也是横图，尝试回贴
                    ratio=h/origin_h# 高固定
                    new_w = int(w / ratio)
                    img_input_new = cv2.resize(img, (new_w, origin_h))  # 原图缩放到输出高度
                    
                    if new_w>origin_w: #原图缩放后宽度超过输出尺寸
                        img_bg=center_crop(img_input_new, origin_w, origin_h) # 中心裁切做背景
                        img_fg = cv2.resize(z, (origin_h, origin_h))  # 缩放到输出高度
                        z =img_coty2_img(img_bg,img_fg) #回帖
                    else: #原图缩放后宽度小于等于输出尺寸
                        
                        img_bg = np.zeros((origin_h,origin_w, 3), np.uint8)
                        img_fg = cv2.resize(img_input_new, (origin_h, origin_h))  # 缩放原图到输出高度
                        img_bg = img_coty2_img(img_bg, img_fg)  # 获得带黑底的背景图
    
                        img_fg = cv2.resize(z, (origin_h, origin_h))  # 缩放实际图到输出高度
                        z = img_coty2_img(img_bg, img_fg)  # 回帖

        z = Image.fromarray(z)

        outputs.append(z)
    #os.makedirs(os.path.dirname(path), exist_ok=True)
    if save_video:
        save_videos_from_pil(outputs, path, int(fps), audio_path=audio_path)
    
    return outputs


def img_coty2_img(img_bg,img_fg):
    h_fg, w_fg = img_fg.shape[:2]
    h_bg, w_bg = img_bg.shape[:2]
    # 计算居中位置
    x = int((w_bg - w_fg) / 2)
    y = int((h_bg - h_fg) / 2)
    # 确保坐标不会是负数
    x = max(0, x)
    y = max(0, y)
    # 确保不会超出大图像的边界
    x = min(x, w_bg - w_fg)
    y = min(y, h_bg - h_fg)
    # 使用NumPy索引将小图像粘贴到大图像上
    img_bg[y:y + h_fg, x:x + w_fg] = img_fg
    return img_bg


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return int(fps)


def crop_and_pad(image, rect):
    x0, y0, x1, y1 = rect
    h, w = image.shape[:2]

    # 确保坐标在图像范围内
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)

    # 计算原始框的宽度和高度
    width = x1 - x0
    height = y1 - y0

    # 使用较小的边长作为裁剪正方形的边长
    side_length = min(width, height)

    # 计算正方形框中心点
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2

    # 重新计算正方形框的坐标
    new_x0 = max(0, center_x - side_length // 2)
    new_y0 = max(0, center_y - side_length // 2)
    new_x1 = min(w, new_x0 + side_length)
    new_y1 = min(h, new_y0 + side_length)

    # 最终裁剪框的尺寸修正（确保是正方形）
    if (new_x1 - new_x0) != (new_y1 - new_y0):
        side_length = min(new_x1 - new_x0, new_y1 - new_y0)
        new_x1 = new_x0 + side_length
        new_y1 = new_y0 + side_length

    # 裁剪图像
    cropped_image = image[new_y0:new_y1, new_x0:new_x1]

    return cropped_image, (new_x0, new_y0, new_x1, new_y1)

def crop_and_pad_rectangle(image,mask, rect,):
    x0, y0, x1, y1 = rect #[89, 6, 334, 363]
    h, w = image.shape[:2] #512，384

    # 确保坐标在图像范围内
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)

    # 裁剪图像
    cropped_image = image[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]
   
    return cropped_image, cropped_mask


def crop_and_pad_rectangle_image(image, rect ):
    x0, y0, x1, y1 = rect  # [89, 6, 334, 363]
    h, w = image.shape[:2]  # 512，384
    
    # 确保坐标在图像范围内
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    
    # 裁剪图像
    cropped_image = image[y0:y1, x0:x1]
    
    return cropped_image

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

def cv2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def tensor_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples

def center_crop(image, crop_width, crop_height):
    # 获取图像的中心坐标
    height, width = image.shape[:2]
    x = width // 2 - crop_width // 2
    y = height // 2 - crop_height // 2
    
    x=max(0,x)
    y=max(0,y)
    
    # 裁剪图像
    crop_img = image[y:y + crop_height, x:x + crop_width]
    return crop_img
