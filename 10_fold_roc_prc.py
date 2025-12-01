import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

#读取 ROC / PRC的pkl文件
def load_roc_prc_data(roc_dir, prc_dir):
    roc_datas, prc_datas = [], []

    roc_files = sorted([f for f in os.listdir(roc_dir) if f.endswith('.pkl')])
    prc_files = sorted([f for f in os.listdir(prc_dir) if f.endswith('.pkl')])

    # 加载 ROC
    for file in roc_files:
        with open(os.path.join(roc_dir, file), 'rb') as f:
            roc_data = pickle.load(f)
            if isinstance(roc_data, dict):
                roc_datas.append((roc_data['fpr'], roc_data['tpr'], roc_data['auc']))
            else:
                roc_datas.append(roc_data)

    # 加载 PRC
    for file in prc_files:
        with open(os.path.join(prc_dir, file), 'rb') as f:
            prc_data = pickle.load(f)
            if isinstance(prc_data, dict):
                prc_datas.append((prc_data['recall'], prc_data['precision'], prc_data['auprc']))
            else:
                prc_datas.append(prc_data)

    return roc_datas, prc_datas


#绘制 ROC 曲线
def ROC_curv(roc_datas):
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []

    for i, (fpr, tpr, auc_value) in enumerate(roc_datas):
        plt.plot(fpr, tpr, lw=1.5, alpha=0.5, label=f'Fold {i + 1} (auROC={auc_value:.3f})')
        # 插值到统一的横坐标上
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    # 计算平均曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean([auc for _, _, auc in roc_datas])
    plt.plot(mean_fpr, mean_tpr, color='black', lw=2.5, linestyle='--',
             label=f'Mean auROC (auROC={mean_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('1 - Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.title('10-Fold cross validation roc curves')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


#绘制 PRC 曲线
def PRC_curv(prc_datas):
    mean_recall = np.linspace(0, 1, 200)
    precisions = []

    for i, (recall, precision, auprc_value) in enumerate(prc_datas):
        plt.plot(recall, precision, lw=1.5, alpha=0.5, label=f'Fold {i + 1} (auPRC={auprc_value:.3f})')
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))

    mean_precision = np.mean(precisions, axis=0)
    mean_auprc = np.mean([auc for _, _, auc in prc_datas])
    plt.plot(mean_recall, mean_precision, color='black', lw=2.5, linestyle='--',
             label=f'Mean auPRC (auPRC={mean_auprc:.3f})')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('10-Fold cross validation prc curves')
    plt.legend(loc='lower left', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    roc_dir = './results/ROC'
    prc_dir = './results/PRC'

    roc_datas, prc_datas = load_roc_prc_data(roc_dir, prc_dir)

    ROC_curv(roc_datas)
    PRC_curv(prc_datas)
