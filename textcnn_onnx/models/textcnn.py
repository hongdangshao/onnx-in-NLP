# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/18 20:22
# @Author  : Shaohd
# @FileName: model.py

import torch
from torch import nn
import torch.nn.functional as F


class TextCnn(nn.Module):
    def __init__(self, config, vocab_size, pretrained_embedding=None):
        super(TextCnn, self).__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding, freeze=config.static_embedding)
        else:
            self.embedding = nn.Embedding(vocab_size, config.hidden_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, c, (k, config.hidden_dim))
            for c, k in zip(config.filter_num, config.kernel_size)])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(sum(config.filter_num), config.class_num)

    def forward(self, x):
        x = self.embedding(x)  # (bs,seq_len,hidden_dim)
        x = x.unsqueeze(1)  # (bs, 1, seq_len,hidden_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(bs, num_filters, seq_len-filter_size+1), ...]
        x = [F.max_pool1d(c, int(c.size(2))).squeeze(2) for c in x]  # [(bs, num_filters), ...]
        x = torch.cat(x, dim=1)  # (bs, sum(num_filters))
        x = self.dropout(x)  # (bs, sum(num_filters))
        logits = self.fc(x)
        return logits
