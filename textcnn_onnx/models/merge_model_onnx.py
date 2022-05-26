# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/26 9:57
# @Author  : Shaohd
# @FileName: merge_model_onnx.py


import torch
import onnxruntime
from torchtext.data.utils import get_tokenizer
import models.datasets as datasets
import numpy as np
from models.config import ModelConfig
from global_config import *
import warnings

warnings.filterwarnings('ignore')

tokenizer = get_tokenizer(datasets.chinese_tokenizer, language='chn')


def char2index(vocab, token):
    try:
        return vocab[token]
    except:
        return 1


class Mergemodel:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vocab_file = os.path.join(model_path, 'vocab.pkl')
        self.vocab = self.load_vocab()
        self.model_file = os.path.join(model_path, 'textcnn.onnx')
        self.model = self.load_model()

    def load_vocab(self):
        return torch.load(open(self.vocab_file, 'rb'))

    def load_model(self):
        return onnxruntime.InferenceSession(self.model_file)

    def predict(self, text):
        text_transform = lambda x: [char2index(self.vocab, token) for token in tokenizer(x)]
        row = np.array([text_transform(text.strip()) + [0] * (ModelConfig().max_seq_len - len(text))], dtype=np.int32)
        ort_output = self.model.run(['output'], {'input': row})
        softmax = torch.nn.Softmax(dim=-1)
        pred = softmax(torch.tensor(ort_output))
        pred = np.argmax(pred.detach().to('cpu').numpy()[0][0])
        return pred


if __name__ == '__main__':
    text = '商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!'
    model = Mergemodel()
    print(model.predict(text))