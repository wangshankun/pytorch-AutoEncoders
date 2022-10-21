#!/usr/bin/env python
# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
import torch.utils.data.dataloader as dataloader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from common.datas import get_mnist_loader
import torch.nn.functional as F

import os
import time
import matplotlib.pyplot as plt
from PIL import Image

batch_size = 100
num_epochs = 30
in_dim = 784
expect_tho = 0.05
#虽然稀疏增加了鲁棒性，但稀疏会导致隐藏层个数不够容纳整个特征，因此隐藏层需要适当扩大节点数量；
hidden_size = int(30 / expect_tho)

def KL_devergence(probs, target, temp = 1):
    lossfunc = nn.KLDivLoss(reduction='batchmean')
    return lossfunc(F.log_softmax(probs / temp, dim=-1), F.softmax(target / temp, dim=-1))

class AutoEncoder(nn.Module):
    def __init__(self, in_dim=784, hidden_size=30, out_dim=784):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out


train_loader, test_loader = get_mnist_loader(batch_size=batch_size, shuffle=True)
testdata_iter = iter(test_loader)
test_images, _ = next(testdata_iter)
if torch.cuda.is_available():
    test_images = test_images.cuda()
torchvision.utils.save_image(test_images, './data/origin_test_images.png')
image_real = Image.open('./data/origin_test_images.png')


autoEncoder = AutoEncoder(in_dim=in_dim, hidden_size=hidden_size, out_dim=in_dim)
if torch.cuda.is_available():
    autoEncoder.cuda()  # 注:将模型放到GPU上,因此后续传入的数据必须也在GPU上

Loss = nn.BCELoss()
Optimizer = optim.Adam(autoEncoder.parameters(), lr=0.001)

# 定义期望平均激活值和KL散度的权重
tho_tensor = torch.FloatTensor([expect_tho for _ in range(hidden_size)])
if torch.cuda.is_available():
    tho_tensor = tho_tensor.cuda()
_beta = 3


for epoch in range(num_epochs):
    time_epoch_start = time.time()
    for batch_index, (train_data, train_label) in enumerate(train_loader):
        if torch.cuda.is_available():
            train_data = train_data.cuda()
            train_label = train_label.cuda()
        input_data = train_data.view(train_data.size(0), -1)
        encoder_out, decoder_out = autoEncoder(input_data)
        loss = Loss(decoder_out, input_data)

        # 计算并增加KL散度到loss
        _kl = KL_devergence(tho_tensor, encoder_out)
        loss += _beta * _kl

        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()

        if (batch_index + 1) % 100 == 0:
            print('Epoch {}/{}, Iter {}/{}, loss: {:.4f}, time: {:.2f}s'.format(
                epoch + 1, num_epochs, (batch_index + 1), len(train_loader), loss, time.time() - time_epoch_start
            ))
    _, val_out = autoEncoder(test_images.view(test_images.size(0), -1).cuda())
    val_out = val_out.view(test_images.size(0), 1, 28, 28)
    filename = './data/reconstruct_sae_images_{}.png'.format(epoch + 1)
    torchvision.utils.save_image(val_out, filename)

