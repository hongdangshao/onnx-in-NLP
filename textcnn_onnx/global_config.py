# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/18 19:19
# @Author  : Shaohd
# @FileName: global_config.py


import os

RELATIVE_PATH = os.path.dirname(__file__)
base_path = os.path.abspath(RELATIVE_PATH)
source_path = os.path.join(base_path, 'source')
data_path = os.path.join(source_path, 'data')
model_path = os.path.join(source_path, 'saved_model')

data_file = os.path.join(data_path, 'CtripSentimentCorp.csv')
emb_path = os.path.join(source_path, 'dict')
word_vec = os.path.join(emb_path, 'sgns.sogou.char')