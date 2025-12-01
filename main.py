import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import random

from sklearn.decomposition import PCA

from model.TCB_Kla import TCB_Kla

from preprocess import loader
from preprocess.loader import data_construct
from termcolor import colored
import pandas as pd
from sklearn.model_selection import KFold
import sys, os, re
import csv


# sys.argv[0]返回当前路径，os.path.realpath（）返回绝对路径，os.path.split()将路径分解为路径和文件名
from util.util_loss import FocalLoss_v2, reg_loss
from util.util_metric import evaluate

pPath = os.path.split(os.path.realpath(sys.argv[0]))[0]
# print(sys.argv[0])
# print(pPath)
sys.path.append(pPath)  # 添加路径
pPath = re.sub(r'codes$', '', os.path.split(os.path.realpath(sys.argv[0]))[0])
sys.path.append(pPath)
import matplotlib


matplotlib.use('TkAgg')  # 使用 Agg 后端（适用于无 GUI 环境）

import numpy as np
import torch
import umap
import os

import torch
from torch import nn

# ========= 日志配置 =========
log_dir = "C:/Users/shengbin/Desktop/ACE-ACP/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)


# ===========================


def train_test(train_iter, test_iter):
    net = TCB_Kla().cuda()
    lr = 0.0002
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_acc = 0
    EPOCH = 50

    loss_list = []

    for epoch in range(EPOCH):
        loss_ls = []
        t0 = time.time()
        net.train()
        for embed_data, pos, label in train_iter:
            embed_data, pos, label = embed_data.cuda(), pos.cuda(), label.cuda()
            # print(f'esm的形状{esm.shape}')
            # alpha = torch.tensor([0.5, 0.5]).cuda()
            _, output = net(embed_data, pos)
            # loss_fn = FocalLoss_v2(num_class=2, gamma=2, alpha=alpha)
            # loss = loss_fn(output, label)

            loss = reg_loss(net, output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())

        avg_loss = np.mean(loss_ls)
        loss_list.append(avg_loss)

        train_performance, train_roc_data, train_prc_data, _, _, _, _ = evaluate(train_iter, net)
        net.eval()
        with torch.no_grad():

            test_performance, test_roc_data, test_prc_data, rep_list, label_real, _, _ = evaluate(test_iter, net)

        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"  # 打印当前训练轮数以及当前轮的平均损失
        results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'  # 打印当前轮训练集的准确率和时间
        results += '\n' + '=' * 16 + ' Test Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC,\tBACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            test_performance[0], test_performance[5], test_performance[2], test_performance[1], test_performance[3],
            test_performance[4]) + '\n' + '=' * 60  # 打印当前轮测试集上的指标

        logger.info(results)  # 保存训练日志

        print(results)
        test_acc = test_performance[0]  # 测试集的准确率  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
        if test_acc > best_acc:
            best_acc = test_acc
            best_performance = test_performance
            filename = '{}, {}[{:.4f}].pt'.format('Model' + ', epoch[{}]'.format(epoch + 1), 'ACC',
                                                  best_acc)  # 保存文件名，比如H_A_Model, epoch[5], ACC[0.8765].pt

            save_path_pt = os.path.join(
                'C:/Users/shengbin/Desktop/TCB-Kla/model_pth',
                filename)
            # torch.save(net.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)

            best_results = '\n' + '=' * 16 + colored(' Best Performance. Epoch[{}] ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tBACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                best_performance[0], best_performance[5], best_performance[2], best_performance[1], best_performance[3],
                best_performance[4]) + '\n' + '=' * 60  # 测试集中预测最好的结果
            # print(best_results)
            best_ROC = test_roc_data
            best_PRC = test_prc_data
    return best_performance, best_results, best_ROC, best_PRC








import os
import pickle

def K_CV(file, k, save_dir_ROC='./results/ROC/', save_dir_PRC='./results/PRC/'):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = np.array(tmp[0].values.tolist()), np.array(tmp[1].values.tolist())
    data_iter = data_construct(seqs, labels, train=True)
    data_iter = list(data_iter)
    CV_perform = []

    #自动创建不存在的目录
    os.makedirs(save_dir_ROC, exist_ok=True)
    os.makedirs(save_dir_PRC, exist_ok=True)

    for iter_k in range(k):
        print("\n" + "=" * 16 + "k = " + str(iter_k + 1) + "=" * 16)
        train_iter = [x for i, x in enumerate(data_iter) if i % k != iter_k]
        test_iter = [x for i, x in enumerate(data_iter) if i % k == iter_k]
        performance, _, ROC, PRC = train_test(train_iter, test_iter)
        print(performance)

        # 保存到指定目录
        with open(os.path.join(save_dir_ROC, f"ROC_{iter_k + 1}.pkl"), "wb") as f:
            pickle.dump(ROC, f)

        with open(os.path.join(save_dir_PRC, f"PRC_{iter_k + 1}.pkl"), "wb") as f:
            pickle.dump(PRC, f)

        CV_perform.append(performance)

    print('\n' + '=' * 16 + colored(' Cross-Validation Performance ', 'red') + '=' * 16 +
          '\n[ACC, \tSP,\t\tSE,\t\tAUC,\tMCC]\n')
    for out in np.array(CV_perform):
        print('{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(out[0], out[2], out[1], out[3], out[4]))

    mean_out = np.array(CV_perform).mean(axis=0)
    print('\n' + '=' * 16 + "Mean out" + '=' * 16)
    print('{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(mean_out[0], mean_out[2], mean_out[1], mean_out[3], mean_out[4]))
    print('\n' + '=' * 60)






if __name__ == '__main__':
    # train_test on benchmark dataset
    file = "C:/Users/shengbin/Desktop/TCB-Kla/dataset/Kla/train.csv"
    train_iter, test_iter = loader.load_bench_data(file)
    _, result_bench, roc_data, prc_data = train_test(train_iter, test_iter)

    #     # # k-fold cross-validation
    # K_CV(file, 10)
