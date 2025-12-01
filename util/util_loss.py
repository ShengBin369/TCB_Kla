import torch
from torch import nn


def reg_loss(net, output, label):
    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()  # 表示对所有的样本损失进行求和
    l2_lambda =0.0001
    regularization_loss = 0
    for param in net.parameters():  # net.parameters()是一个生成器，它会返回列表中所有可训练的参数，param是模型中的参数，通常为tensor张量+
        regularization_loss += torch.norm(param, p=2)

    total_loss = criterion(output, label) + l2_lambda * regularization_loss
    return total_loss

# def reg_loss(net, output, label):
#     """
#     计算带有类别权重和L2正则化的损失函数
#     """
#     # ====== 1️⃣ 设置类别权重（根据类别比例调整） ======
#     # 假设类别0（负样本）数量远多于类别1（正样本）
#     # 权重一般取反比于样本数，例如：权重 = [1, 8]  表示正样本权重大8倍
#     weight = torch.tensor([1.0, 8.0]).cuda()
#
#     # ====== 2️⃣ 定义带权重的交叉熵损失 ======
#     criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean').cuda()
#
#     # ====== 3️⃣ 计算L2正则项 ======
#     l2_lambda = 0.0001
#     regularization_loss = torch.tensor(0., device=output.device)
#     for param in net.parameters():
#         regularization_loss += torch.norm(param, p=2)
#
#     # ====== 4️⃣ 计算总损失 ======
#     total_loss = criterion(output, label) + l2_lambda * regularization_loss
#
#     return total_loss

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#used
class FocalLoss_v2(nn.Module):
    def __init__(self, num_class=2, gamma=2, alpha=None):

        super(FocalLoss_v2, self).__init__()
        self.gamma = gamma
        self.num_class = num_class
        if alpha == None:
            self.alpha = torch.ones(num_class)
        else:
            self.alpha=alpha

    def forward(self, logit, target):

        target = target.view(-1)

        alpha = self.alpha[target.cpu().long()]

        logpt = - F.cross_entropy(logit, target, reduction='none')
        pt = torch.exp(logpt)
        focal_loss = -(alpha * (1 - pt) ** self.gamma) * logpt

        return focal_loss.mean()