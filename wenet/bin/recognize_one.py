#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function


import os 
import sys
sys.path.append(os.path.dirname(__file__))
import torch
import yaml
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint

def process_wav(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(waveform,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                energy_floor=0.0,
                sample_frequency=16000)
    feat = torch.unsqueeze(mat, 0)
    hyps, _ = model.recognize(
                        feat,
                        torch.tensor([mat.shape[0]]),
                        beam_size=10,
                        decoding_chunk_size=-1,
                        num_decoding_left_chunks=-1)
    hyps = [hyp.tolist() for hyp in hyps]
    content = ''
    for w in hyps[0]:
        if w == eos:
            break
        content += char_dict[w]
    return content

config_path = './20220506_u2pp_conformer_exp/train.yaml'
with open(config_path, 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)

model = init_asr_model(configs)
dict_path = './20220506_u2pp_conformer_exp/words.txt'
# Load dict
char_dict = {}
with open(dict_path, 'r', encoding="utf-8") as fin:
    for line in fin:
        arr = line.strip().split()
        assert len(arr) == 2
        char_dict[int(arr[1])] = arr[0]
eos = len(char_dict) - 1
checkpoint_path = './20220506_u2pp_conformer_exp/final.pt'
load_checkpoint(model, checkpoint_path)


logfile=open('./errorLog.txt','w+')
for file in os.listdir('./asrfile'):
    print("file：" + file)
    dir = './asrfile/' + file
    path = './res_file/res_' + file + '.txt'
    print("dir:" + dir)
    print("path" + path)
    if(os.path.isfile(path)):
        print(path + "已跑过，跳转下一个")
        continue
    f = open(path,'w+')
    for filefile in os.listdir(dir):
        try:
            print("filefile:" + filefile)
            filefilepath = dir + '/' + filefile
            f.write("{0} {1}\n".format(filefile, process_wav(filefilepath)))
        except:
            RuntimeError
            logfile.write("音频文件:" + file + "的第" + filefile + "出了问题")


    f.close()
logfile.close()
# wav_file = './asr0.wav'
# print(process_wav(wav_file))