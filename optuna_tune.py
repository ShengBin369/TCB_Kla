import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import time, os
from model.TCB_Kla import TCB_Kla
from util.util_metric import evaluate
from util.util_loss import reg_loss
from preprocess.loader import load_bench_data
import logging


# ===================== logger 设置 =====================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("optuna_logger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(log_dir, "optuna_tuning.log"))
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())


# ===================== 训练与评估函数 =====================
def objective(trial):
    # ------ 1. 采样超参数 ------
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    dropout = trial.suggest_float('dropout', 0.3, 0.7)
    conv_channels = trial.suggest_categorical('conv_channels', [32, 64, 128])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_heads = trial.suggest_categorical('num_heads', [4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 3)

    # ------ 2. 数据加载 ------
    file = "C:/Users/shengbin/Desktop/TCB-Kla/dataset/Kla/train.csv"  # 两列，第一列是氨基酸序列，第二列是标签
    train_iter, test_iter = load_bench_data(file)

    # ------ 3. 构建模型 ------
    model = TCB_Kla(
        d_model=128,
        num_heads=num_heads,
        max_len=45,
        num_encoder_layers=num_layers,
        conv_channels=conv_channels,
        dropout=dropout,
        hidden_dim=hidden_dim
    ).cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------ 4. 训练模型 ------
    EPOCH = 15  # 每个 trial 只训练少量 epoch，加快搜索
    best_acc = 0.0

    for epoch in range(EPOCH):
        model.train()
        losses = []
        for embed_data, pos, label in train_iter:
            embed_data, pos, label = embed_data.cuda(), pos.cuda(), label.cuda()
            _, output = model(embed_data, pos)
            loss = reg_loss(model, output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        _, _, _, _, _, _, _ = evaluate(train_iter, model)
        model.eval()
        with torch.no_grad():
            test_perf, _, _, _, _, _, _ = evaluate(test_iter, model)
        test_acc = test_perf[0]

        logger.info(f"[Trial {trial.number}] Epoch {epoch+1}/{EPOCH} | Loss: {avg_loss:.4f} | Test_ACC: {test_acc:.4f}")

        # Early stopping
        trial.report(test_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if test_acc > best_acc:
            best_acc = test_acc

    return best_acc



if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=20)  # 调20组超参数

    logger.info("Best trial:")
    logger.info(f"  Value: {study.best_trial.value}")
    logger.info(f"  Params: {study.best_trial.params}")

    # ========== 可视化 ==========
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()
    # 平行坐标图（展示不同参数组合与结果的关系）
    optuna.visualization.plot_parallel_coordinate(study).show()

    # 参数关系图（查看两个参数之间的交互对性能的影响）
    optuna.visualization.plot_contour(study).show()

    # 超参数采样分布（查看每个参数被采样的数值分布）
    optuna.visualization.plot_slice(study).show()
