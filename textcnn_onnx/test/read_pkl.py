# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/19 9:41
# @Author  : Shaohd
# @FileName: read_pkl.py

import torch
import os
from global_config import model_path

f = open(os.path.join(model_path, 'vocab.pkl'), 'rb')
vocab = torch.load(f, map_location='cpu')

print(f"vocab:\n{vocab}")