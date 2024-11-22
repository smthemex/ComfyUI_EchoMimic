#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：EMO_digitalhuman 
@File    ：wav_clip.py
@Author  ：juzhen.czy
@Date    ：2024/3/4 19:04 
'''
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch
from torch import nn
import librosa
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange, repeat


class Wav2Vec(ModelMixin):
    def __init__(self, model_path):
        super(Wav2Vec, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.wav2Vec = Wav2Vec2Model.from_pretrained(model_path)
        self.wav2Vec.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.wav2Vec(x).last_hidden_state

    # def forward(self, x):
    #     return self.wav2Vec(x).last_hidden_state

    def process(self, x):
        return self.processor(x, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)

class AudioFeatureMapper(ModelMixin):
    def __init__(self, input_num=15, output_num=77, model_path=None):
        super(AudioFeatureMapper, self).__init__()
        self.linear = nn.Linear(input_num, output_num)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path))

    def forward(self, x):
        # print(x.shape)
        result = self.linear(x.permute(0, 2, 1))
        result = result.permute(0, 2, 1)
        # result = self.linear(x)
        return result

def test():
    #加载模型
    model_path = "/ossfs/workspace/projects/model_weights/Moore-AnimateAnyone/wav2vec2-base-960h"
    model = Wav2Vec(model_path)
    print("### model loaded ###")
    #加载音频
    audio_path = "/ossfs/workspace/projects/Moore-AnimateAnyone-master/assets/taken_clip.wav"
    input_audio, rate = librosa.load(audio_path, sr=16000)
    print(f"输入shape: {input_audio.shape}, rate: {rate}")

    # 预处理, 维度变为 (1, input_audio.shape[0]), 增加了一个维度, 声音信号长度本身没有变
    input_v = model.process(input_audio)

    # 输出结果为
    out = model(input_v)
    print(f"输入shape: {input_v.shape}, 输出shape: {out.shape}")