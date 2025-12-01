import pickle

# import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")



import numpy as np
import matplotlib.pyplot as plt
import pickle

def ROC_curv(roc_datas, labels):
    for i, (fpr, tpr, auc_value) in enumerate(roc_datas):
        plt.plot(fpr, tpr, lw=2, label=f'{labels[i]} (auROC = {auc_value:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('1 - Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.title('ROC curve')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def PRC_curv(prc_datas, labels):
    for i, (recall, precision, auprc_value) in enumerate(prc_datas):
        plt.plot(recall, precision, lw=2, label=f'{labels[i]} (auPRC = {auprc_value:.4f})')

    plt.plot([0, 1], [1, 0], 'k--', lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('PR curve')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# === 加载 ROC 数据 ===
roc_filenames = [
    "train_Kla_ROC.pkl",
    "test_Kla_ROC.pkl"
]

prc_filenames = [
    "train_Kla_PRC.pkl",
    "test_Kla_PRC.pkl"
]

labels = ["train dataset", "independent test set"]

roc_datas = []
for file in roc_filenames:
    with open(file, "rb") as f:
        roc_data = pickle.load(f)  # (fpr, tpr, auc)
        roc_datas.append(roc_data)

prc_datas = []
for file in prc_filenames:
    with open(file, "rb") as f:
        prc_data = pickle.load(f)  # (recall, precision, auprc)
        prc_datas.append(prc_data)

# === 绘图 ===
ROC_curv(roc_datas, labels)
PRC_curv(prc_datas, labels)