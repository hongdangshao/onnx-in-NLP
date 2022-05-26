# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 13:49
# @Author  : Shaohd
# @FileName: eda.py

import numpy as np
import pandas as pd
from global_config import *

annotations = pd.read_csv(data_file, header=0)
len_lst = []
l = len(annotations)

for idx in range(l):
    text = annotations.iloc[idx, 1]
    len_lst.append(len(text))

for percentile in range(0, 101, 10):
    print(f"{percentile:>4}% 分位数正文文本长度为：{np.percentile(len_lst, percentile)}")