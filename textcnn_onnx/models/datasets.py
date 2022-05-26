# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/18 20:22
# @Author  : Shaohd
# @FileName: datasets.py


from collections import Counter, OrderedDict
import pandas as pd
from torch.utils.data import Dataset


class CtripSentimentDataset(Dataset):
    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file, header=0)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        label = self.annotations.iloc[idx, 0]
        text = self.annotations.iloc[idx, 1]
        return text, label


def chinese_tokenizer(text):
    # char level
    return [tok for tok in text]


def build_vocab(dataset, tokenizer):
    counter = Counter()
    for text, label in dataset:
        counter.update(tokenizer(text))
    sorted_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    sorted_tuples = [i for i in sorted_tuples if i[1] >= 1]
    ordered_dict = OrderedDict(sorted_tuples)

    vocab = dict()
    pad_token = '<PAD>'
    unk_token = '<UNK>'
    vocab[pad_token] = 0
    vocab[unk_token] = 1
    i = 2
    for key in ordered_dict.keys():
        vocab[key] = i
        i += 1
    return vocab