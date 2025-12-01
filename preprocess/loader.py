import pandas as pd
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
# 编码过程以及数据加载过程
import torch

from imblearn.over_sampling import SMOTE
import numpy as np

# 对每个序列中的氨基酸进行序列编码，编码维度为128
def position_encoding(seqs):
    """
    Position encoding features introduced in "Attention is all your need",
    the b is changed to 1000 for the short length of sequence.
    """
    d = 128
    b = 1000
    res = []
    for seq in seqs:
        N = len(seq)
        value = []
        for pos in range(N):
            tmp = []
            for i in range(d // 2):
                tmp.append(pos / (b ** (2 * i / d)))
            value.append(tmp)
        value = np.array(value)
        pos_encoding = np.zeros((N, d))
        pos_encoding[:, 0::2] = np.sin(value[:, :])
        pos_encoding[:, 1::2] = np.cos(value[:, :])
        res.append(pos_encoding)  # 其中pos_encoding偶数列为sin,奇数列为cos,维度为（seq_len,d）
    return np.array(res)  # 最终res的位置编码维度为 (batch_size, seq_len, d),数据类型为numpy数组


def data_construct(seqs, labels, train):  # 参数seqs表示未经过填充的原始序列，sequences表示填充后的序列

    longest_num = 45
    seqs = [i.ljust(longest_num, 'X') for i in seqs]  # 列表推导式i.ljust(a,b)表示对i进行b填充到长度a
    # 氨基酸字典
    aa_dict = {'X': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22}
    pos_embed = position_encoding(seqs)  # sequences形式为列表

    pep_codes = []  # 用于储存氨基酸序列的编码
    for pep in seqs:  # 序列列表进行循环
        current_pep = []
        for aa in pep:  # 对单个序列进行循环
            current_pep.append(aa_dict[aa.upper()])  # 查找 aa 在 aa_dict 中的编码值并将其加入 current_pep
        pep_codes.append((torch.tensor(current_pep)))  # 将当前序列的编码（列表）转化为 tensor 并添加到 pep_codes

    # pad_sequence会自动用0进行填充，使所有序列的长度与最长的序列保持一致。batch_first表示把batch_size放在第一个维度（batch_size,seq_max_len）
    embed_data = rnn_utils.pad_sequence(pep_codes, batch_first=True).cuda()
    # 数据集封装类，用于将多个tensor组装成一个数据集，氨基酸序列的嵌入表示，位置编码，手工特征，标签
    dataset = Data.TensorDataset(embed_data, torch.tensor(pos_embed).cuda(),
                                 torch.LongTensor(labels).cuda())  # LongTensor将数据转换为64位int类型
    batch_size = 128
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return data_iter


# 原来最初的加载数据的形式
def load_bench_data(file):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, labels, train=True)
    data_iter = list(data_iter)
    train_iter = [x for i, x in enumerate(data_iter) if i % 5 != 0]
    test_iter = [x for i, x in enumerate(data_iter) if i % 5 == 0]

    return train_iter, test_iter


def load_ind_data(file):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, labels,
                               train=False)  # 一个用于将氨基酸序列和标签数据转换为适合神经网络输入的格式并生成数据迭代器的函数。
    return data_iter


def load_model(new_model, path_pretrain_model):
    pretrained_dict = torch.load(path_pretrain_model, map_location=torch.device('cpu'))
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)
    return new_model


if __name__ == "__main__":
    file = "C:/Users/shengbin/Desktop/抗糖尿病肽（验证）/dataset/Kla/train.csv"  # 两列，第一列是氨基酸序列，第二列是标签
    esm_file = "C:/Users/shengbin/Desktop/ACE-ACP/dataset/ACE/esm2_8M_feature/esm_train.csv"
    train_iter, test_iter = load_bench_data(file, esm_file)
    print("Train Set:")
    for batch in train_iter[:1]:  # 只看一个 batch 示例
        embed_data, pos_embed, labels = batch
        print(f"esm_tensor.shape:       {embed_data.shape}       # trans_former编码提取的特征(batch_size, seq_len, 128)")
        print(f"pos_embde.shape:  {pos_embed.shape}  #  位置编码的特征 (batch_size, seq_len, 1280)")
        # print(f"esm.shape:  {esm.shape}  #  位置编码的特征 (batch_size, seq_len, 1280)")
        print(f"labels.shape:          {labels.shape}          # 标签向量 (batch_size,)")

        # 打印第一个样本的数据
        first_embed = embed_data[0]  # 取第一个样本
        first_label = labels[0]  # 对应的标签

        print("\n第一个样本的嵌入特征张量：")
        print(first_embed)  # 这是一个 [seq_len, feature_dim] 的张量

        print("\n第一个样本的标签：")
        print(first_label)

    print("\nTest Set:")
    for batch in test_iter[:1]:
        embed_data,pos_embed, labels = batch
        print(f"esm_tensor.shape:       {embed_data.shape}")
        print(f"pos_embde.shape:  {pos_embed.shape}")  # one-hot 提取的特征 (batch_size, seq_len, 1280)
        # print(f"esm.shape:  {esm.shape}")  # esm 提取的特征 (batch_size, seq_len, 1280)
        print(f"labels.shape:  {labels.shape}")
    for batch in train_iter:
        labels = batch[-1].cpu().numpy()
        print(labels)
        break
