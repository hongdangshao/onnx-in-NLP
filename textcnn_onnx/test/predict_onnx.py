# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 9:12
# @Author  : Shaohd
# @FileName: predict_onnx.py

import torch
import onnx
import onnxruntime
from torchtext.data.utils import get_tokenizer
import models.datasets as datasets
import numpy as np
from models.config import ModelConfig
from global_config import *
import warnings

warnings.filterwarnings('ignore')

model = onnx.load(os.path.join(model_path, 'textcnn.onnx'))
onnx.checker.check_model(model)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
f = open(os.path.join(model_path, 'vocab.pkl'), 'rb')
vocab = torch.load(f, map_location='cpu')

ort_sess = onnxruntime.InferenceSession(os.path.join(model_path, 'textcnn.onnx'))

tokenizer = get_tokenizer(datasets.chinese_tokenizer, language='chn')

def char2index(vocab, token):
    try:
        return vocab[token]
    except:
        return 1

text_transform = lambda x: [char2index(vocab, token) for token in tokenizer(x)]

import time
s = time.time()
n = 1000
for _ in range(n):
    row = '商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!'
    row = '酒店卫生和环境太差了'
    row = np.array([text_transform(row.strip()) + [0] * (ModelConfig().max_seq_len-len(row))], dtype=np.int32)

    ort_output = ort_sess.run(['output'], {'input': row})

    softmax = torch.nn.Softmax(dim=-1)
    pred = softmax(torch.tensor(ort_output))
    pred = pred.detach().to('cpu').numpy()
    print(f"predict: {pred[0]}")

e = time.time()
print(f"Every text elapsed {1000*(e-s)/n:.3f}ms")   # 1.656ms,我去，确实提速明显，速度提升超过 50%