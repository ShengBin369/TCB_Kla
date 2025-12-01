import logging
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from model.TCB_Kla import TCB_Kla
from preprocess.loader import load_ind_data, load_model
from termcolor import colored
from util.util_metric import evaluate
import matplotlib

matplotlib.use('TkAgg')
#日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
    handlers=[
        logging.StreamHandler(),  # 打印到控制台
        logging.FileHandler("C:/Users/shengbin/Desktop/TCB-Kla/logs/log.txt", mode="w", encoding="utf-8")  # 保存到文件
    ]
)
logger = logging.getLogger(__name__)


def predict(file):
    logger.info(f"Loading independent data from: {file}")
    data_iter = load_ind_data(file)
    model = TCB_Kla().cuda()
    path_pretrain_model = "model_pth/TCB_Kla.pt"
    model = load_model(model, path_pretrain_model)
    model.eval()
    logger.info("Start evaluating on independent test set...")
    with torch.no_grad():
        ind_performance, ind_roc_data, ind_prc_data, rep_list, label_real, label_pred, pred_prob = evaluate(data_iter,
                                                                                                            model)
        #可视化部分
        rep_array = np.array(rep_list)
        labels = np.array(label_real)

        logger.info("Performing t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        rep_2d = tsne.fit_transform(rep_array)

        #绘制散点图，直接在图中显示Kla/Non-Kla
        plt.figure(figsize=(8, 6))
        colors = ['#1f77b4', '#ff7f0e']  # 0->non-Kla 蓝色, 1->Kla 橙色
        for label_value, label_name, color in zip([0, 1], ['non-Kla', 'Kla'], colors):
            idx = labels == label_value
            plt.scatter(rep_2d[idx, 0], rep_2d[idx, 1], c=color, label=label_name, alpha=0.7)

        plt.title("T-SNE visualization on independent test set", fontsize=14)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(title=None)
        plt.tight_layout()

        vis_save_path = "C:/Users/shengbin/Desktop/TCB-Kla/Ind_tsne_picture/ind.png"
        plt.savefig(vis_save_path, dpi=300)
        plt.close()
        logger.info(f"t-SNE visualization saved to: {vis_save_path}")
    ind_results = '\n' + '=' * 16 + colored(' Independent Test Performance', 'red') + '=' * 16 \
                  + '\n[ACC,\tBACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
        ind_performance[0], ind_performance[5], ind_performance[2], ind_performance[1], ind_performance[3],
        ind_performance[4]) + '\n' + '=' * 60
    logger.info("Evaluation finished.")
    logger.info(
        f"Performance: ACC={ind_performance[0]:.4f},BACC={ind_performance[5]:.4f}, SP={ind_performance[2]:.4f}, "
        f"SE={ind_performance[1]:.4f}, AUC={ind_performance[3]:.4f}, MCC={ind_performance[4]:.4f}")

    return ind_results


file = 'C:/Users/shengbin/Desktop/TCB-Kla/dataset/Kla/test.csv'
ind_result = predict(file)
print(ind_result)

