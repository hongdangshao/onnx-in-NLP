# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/18 19:19
# @Author  : Shaohd
# @FileName: config.py


class ModelConfig(object):
    def __init__(self):
        self.hidden_dim = 300
        self.static_embedding = False
        self.kernel_size = [3, 4, 5]
        self.filter_num = [200, 200, 200]
        self.dropout = 0.5
        self.class_num = 2
        self.max_seq_len = 256


class TrainConfig(object):
    def __init__(self):
        self.learning_rate = 1e-3
        self.batch_size = 32
        self.epochs = 10
        self.shuffle = True
        self.train_size = 0.8