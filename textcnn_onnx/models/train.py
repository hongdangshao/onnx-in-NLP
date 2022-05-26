# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/18 20:22
# @Author  : Shaohd
# @FileName: train.py


import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

from models.textcnn import TextCnn
from config import ModelConfig, TrainConfig
import datasets

from global_config import *


model_config = ModelConfig()
train_config = TrainConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = datasets.CtripSentimentDataset(data_file)

train_size = int(train_config.train_size * len(dataset))
test_size = len(dataset) - train_size
training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

tokenizer = get_tokenizer(datasets.chinese_tokenizer, language='chn')
vocab = datasets.build_vocab(training_data, tokenizer)


def save_model(model, save_dir, filename):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    torch.save(model, save_path)

save_model(vocab, model_path, 'vocab.pkl')

def char2index(vocab, token):
    try:
        return vocab[token]
    except:
        return 1

text_transform = lambda x: [char2index(vocab, token) for token in tokenizer(x)]
label_transform = lambda x: int(x)


def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        _text = _text[:model_config.max_seq_len]
        processed_text = torch.tensor(text_transform(_text))
        text_list.append(processed_text)
        label_list.append(label_transform(_label))
    label_list = torch.tensor(label_list)
    return pad_sequence(text_list, batch_first=True).to(device), label_list.to(device)


train_dataloader = DataLoader(training_data,
                              batch_size=train_config.batch_size,
                              shuffle=train_config.shuffle,
                              collate_fn=collate_batch)

test_dataloader = DataLoader(test_data,
                             batch_size=train_config.batch_size,
                             shuffle=train_config.shuffle,
                             collate_fn=collate_batch)


model = TextCnn(model_config, len(vocab), None).to(device)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if batch > 0 and batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    print(f'test error:\nAccuracy: {(100 * accuracy):>0.3f}%, Avg loss: {test_loss:>8f}\n')
    return accuracy


CE_Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=train_config.learning_rate)

best_accu = 0
for t in range(train_config.epochs):
    print(f"Epoch {t + 1}\n")
    train_loop(train_dataloader, model, CE_Loss, optimizer)
    accu = test_loop(test_dataloader, model, CE_Loss)
    if accu > best_accu:
        best_accu = accu
        save_model(model, model_path, 'best_model.pth')
        print(f'Best accuracy: {best_accu}\n')
print(f'Global best accuracy: {best_accu}')