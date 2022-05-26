# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/18 20:22
# @Author  : Shaohd
# @FileName: predict.py


import torch
from torchtext.data.utils import get_tokenizer
import models.datasets as datasets
from models.config import ModelConfig
from global_config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

f = open(os.path.join(model_path, 'vocab.pkl'), 'rb')
vocab = torch.load(f, map_location='cpu')

model = torch.load(os.path.join(model_path, 'best_model.pth'), map_location=torch.device('cpu'))
model.eval()

tokenizer = get_tokenizer(datasets.chinese_tokenizer, language='chn')

def char2index(vocab, token):
    try:
        return vocab[token]
    except:
        return 1

text_transform = lambda x: [char2index(vocab, token) for token in tokenizer(x)]
softmax = torch.nn.Softmax(dim=1)

import time
s = time.time()
n = 1000
for _ in range(n):
    row = '商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!'
    row = '酒店卫生和环境太差了'
    row = torch.tensor(text_transform(row.strip()) + [0]*(ModelConfig().max_seq_len-len(row))).to(device)
    row = torch.unsqueeze(row, 0)  # (1, seq_len)
    pred = model(row)  # (1, class_num)
    pred = softmax(pred)
    pred = pred.detach().to('cpu').numpy()
    print(f"predict: {pred}")

e = time.time()
print(f"Every text elapsed {1000*(e-s)/n:.3f}ms")   # 3.984ms