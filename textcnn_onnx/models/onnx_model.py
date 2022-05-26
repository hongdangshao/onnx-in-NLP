# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/18 19:27
# @Author  : Shaohd
# @FileName: onnx_model.py

import torch
import warnings
from config import ModelConfig

from global_config import *

warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load(os.path.join(model_path, 'best_model.pth'), map_location=torch.device('cpu'))
model.eval()
print(f"model structure:\n{model}")

dummy_input = torch.ones([1, ModelConfig().max_seq_len], dtype=torch.int, device=device)
print(dummy_input.shape)

# 固定长度模型导出并可成功推理
torch.onnx.export(model, dummy_input, os.path.join(model_path, 'textcnn.onnx'), verbose=True,
                  opset_version=11, input_names=['input'], output_names=['output'])


# 可变长度模型可成功导出并可成功推理(只有在训练前进行文本数据的max_seq_len处理才可用，一开始没有注意设置文本最大长度，导致可变模型一直报错！！)
# 文本长度可变
torch.onnx.export(model, dummy_input, os.path.join(model_path, 'textcnn_dynamic.onnx'), verbose=True,
                  opset_version=11, input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {1: 'seq_len'}})
# 数据量，文本长度可变
torch.onnx.export(model, dummy_input, os.path.join(model_path, 'textcnn_dynamic_1.onnx'), verbose=True,
                  opset_version=11, input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size', 1: 'seq_len'}, 'output': {0: 'batch_size'}})


"""
小结：
    1.文本模型.pt转换成.onnx模型时，需要首先确定文本的最大长度max_seq_len，在数据预处理时和onnx模型转换时都需要用到，
    确定最大文本长度后，模型可以成功导出并成功推理，推理结果和pt模型基本一致(存在估计精度误差，大约10^-4级别)
    2.当采用不定长的文本模型转onnx模型时，模型可以转换并成功推理，
    但如果训练过程中没有设置文本的最大长度max_seq_len,当用ort推理时，会报错！！——坑点
"""