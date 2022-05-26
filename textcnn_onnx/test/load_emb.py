# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 17:58
# @Author  : Shaohd
# @FileName: load_emb.py

import numpy as np
from models.config import ModelConfig
from global_config import *


def load_pretrained_embedding(word_vec, emb_dim):
    w2v = {}
    with open(word_vec, 'r', encoding='utf-8') as f:
        for idx, row in enumerate(f):
            if idx == 0:
                pass
            toks = row.strip().split(' ')
            w2v[toks[0]] = np.array(list(map(float, toks[-emb_dim:])))
    return w2v


if __name__ == '__main__':
    pretrained_embedding = load_pretrained_embedding(word_vec, ModelConfig().hidden_dim)
    print(f"total words:{len(pretrained_embedding)}")