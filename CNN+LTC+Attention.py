#!/usr/bin/env python3
"""
CNN + 全长时序 + 多头注意力模型
辣椒气体数据集（7类）训练脚本。
每个.xlsx文件对应一个样本。行=时间，列=传感器（10个）。
截取至T=116（最短时间）。分层K折交叉验证。

将每折最佳模型保存至OUTPUT_DIR。

"""

import os
import glob
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----------------------------
# 超参数定义
# ----------------------------
DATA_ROOT = "/home/SunY/Data/lj_data3/lj_data3/C"  # 辣椒数据路径
EXAMPLE_FILE = "/home/SunY/Data/lj_data3/lj_data3/C/0/C1_1.xlsx"
OUTPUT_DIR = "/home/SunY/Code/Demo/result"               # 权重输出路径
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIXED_LEN = 116        # 截断至固定长度（保证所有样本统一长度，避免因时间步数不一致而导致模型输入不匹配）
NUM_CLASSES = 7        # 类别数
SEED = 42              # 随机种子
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

# 训练参数
K_FOLDS = 5       # K折交叉验证
EPOCHS = 60       # 每折最大训练轮数
BATCH_SIZE = 8    # 批大小
LR = 1e-3         # 学习率
PATIENCE = 8      # 早停耐心值
CNN_FEAT = 64     # CNN特征维度
LTC_HIDDEN = 128  # LTC隐藏单元数
MHA_HEADS = 4     # 多头注意力头数
WEIGHT_DECAY = 1e-5   # 权重衰减

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------
# 工具函数：数据集和数据处理
# ----------------------------
# 收集所有文件路径和标签
def collect_files_labels(root_dir):
    root = Path(root_dir)
    files = []    # 所有文件路径
    labels = []   # 对应标签
    for c in range(NUM_CLASSES):  # 类别文件夹0~6
        folder = root / str(c)    
        if not folder.exists():
            print(f"[WARN] folder {folder} not found")
            continue
        for p in sorted(folder.glob("*.xlsx")):  # 遍历所有.xlsx文件
            files.append(str(p))
            labels.append(int(c))
    files = np.array(files)     # 转为numpy数组
    labels = np.array(labels)   
    return files, labels

# 自定义Dataset，读取.xlsx文件，截断/填充至固定长度
class GasDataset(Dataset):
    """文件和标签均为长度相等的数组（列表）。
       每个元素读取一个 .xlsx 文件，截断/补齐至 FIXED_LEN，返回 (T,F) 型 numpy 数组。"""
    def __init__(self, files, labels, fixed_len=FIXED_LEN, transform=None):
        self.files = list(files)
        self.labels = list(labels)
        self.fixed_len = int(fixed_len)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        lbl = int(self.labels[idx])
        # read excel robustly
        try:
            df = pd.read_excel(p, engine='openpyxl')
        except Exception:
            df = pd.read_excel(p)
        arr = df.values.astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # truncate or pad
        if arr.shape[0] >= self.fixed_len:
            arr = arr[:self.fixed_len, :]
        else:
            pad = np.zeros((self.fixed_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])
        if self.transform is not None:
            arr = self.transform(arr)
        return arr, lbl, p

#
def collate_fixed(batch):
    # 批次：列表 (arr(T,F), 标签, 路径)
    seqs = [torch.from_numpy(x[0]) for x in batch]  # each (T,F)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    paths = [x[2] for x in batch]   #文件路径
    data = torch.stack(seqs, dim=0)  # (B, T, F)
    data = data.permute(0, 2, 1).contiguous()  # (B, F, T) for Conv1d
    return data, labels, paths


# ----------------------------
# 完整LTC实现（Liquid Time Constant RNN）建模时序动态
#LTC 是一个神经 ODE-RNN    每一步迭代求解神经微分方程   比普通 RNN 更适合物理信号
# ----------------------------

# ODE求解器类型
class ODESolver:
    SemiImplicit = 'semi'  # 半隐式
    Explicit = 'exp'       # 显式
    RungeKutta = 'rk4'     # 龙格-库塔4阶            

# LTC单元
class LTCCell(nn.Module):
    """详细LTC单元，半隐式求解器"""
    def __init__(self, input_size, hidden_size,
                 solver=ODESolver.SemiImplicit,
                 ode_unfolds=6,
                 input_mapping='affine',
                 erev_init_factor=1.0,
                 w_init_min=0.01, w_init_max=1.0,
                 cm_init=0.5, gleak_init=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.solver = solver
        self.ode_unfolds = ode_unfolds
        self.input_mapping = input_mapping

        # 输入映射参数
        if input_mapping in ('affine', 'linear'):
            self.input_w = nn.Parameter(torch.ones(input_size))
            self.input_b = nn.Parameter(torch.zeros(input_size)) if input_mapping == 'affine' else None
        else:
            self.input_w = None
            self.input_b = None

        # 感官突触 (I -> H)
        self.sensory_W = nn.Parameter(torch.empty(input_size, hidden_size).uniform_(w_init_min, w_init_max))
        self.sensory_mu = nn.Parameter(torch.empty(input_size, hidden_size).uniform_(0.3, 0.8))
        self.sensory_sigma = nn.Parameter(torch.empty(input_size, hidden_size).uniform_(3.0, 8.0))
        self.sensory_erev = nn.Parameter((torch.randint(0, 2, (input_size, hidden_size)).float() * 2 - 1) * erev_init_factor)

        # 递归突触 (H -> H)
        self.W = nn.Parameter(torch.empty(hidden_size, hidden_size).uniform_(w_init_min, w_init_max))
        self.mu = nn.Parameter(torch.empty(hidden_size, hidden_size).uniform_(0.3, 0.8))
        self.sigma = nn.Parameter(torch.empty(hidden_size, hidden_size).uniform_(3.0, 8.0))
        self.erev = nn.Parameter((torch.randint(0, 2, (hidden_size, hidden_size)).float() * 2 - 1) * erev_init_factor)

        # 神经元参数
        self.vleak = nn.Parameter(torch.empty(hidden_size).uniform_(-0.2, 0.2))
        self.gleak = nn.Parameter(torch.full((hidden_size,), gleak_init))
        self.cm_t = nn.Parameter(torch.full((hidden_size,), cm_init))

        # 参数约束范围
        self.w_min = 1e-6; self.w_max = 1e3
        self.gleak_min = 1e-6; self.gleak_max = 1e3
        self.cm_min = 1e-6; self.cm_max = 1e3

    # 输入映射
    def _map_inputs(self, x):
        # x: (B, I)
        if self.input_w is not None:
            x = x * self.input_w
        if self.input_b is not None:
            x = x + self.input_b
        return x

    # 参数约束
    def clamp_parameters(self):
        with torch.no_grad():
            self.W.clamp_(self.w_min, self.w_max)
            self.sensory_W.clamp_(self.w_min, self.w_max)
            self.gleak.clamp_(self.gleak_min, self.gleak_max)
            self.cm_t.clamp_(self.cm_min, self.cm_max)

    # 前向传播
    def forward(self, inputs, state):
        # inputs: (B, I), state: (B, H)
        if self.input_mapping in ('affine', 'linear'):
            inputs = self._map_inputs(inputs)
        if self.solver == ODESolver.SemiImplicit:
            return self._ode_step_semi(inputs, state)
        elif self.solver == ODESolver.RungeKutta:
            return self._ode_step_rk4(inputs, state)
        else:
            return self._ode_step_explicit(inputs, state)

    # 半隐式ODE求解
    def _ode_step_semi(self, inputs, state):
        # semi-implicit integration
        v_pre = state  # (B, H)
        inp = inputs.unsqueeze(-1)  # (B, I, 1)
        sensory_sig = torch.sigmoid(self.sensory_sigma.unsqueeze(0) * (inp - self.sensory_mu.unsqueeze(0)))  # (B, I, H)
        sensory_w_activation = self.sensory_W.unsqueeze(0) * sensory_sig  # (B, I, H)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev.unsqueeze(0)  # (B, I, H)
        w_numerator_sensory = sensory_rev_activation.sum(dim=1)  # (B, H)
        w_denominator_sensory = sensory_w_activation.sum(dim=1)  # (B, H)

        for _ in range(self.ode_unfolds):
            v = v_pre.unsqueeze(-1)  # (B, H, 1)
            rec_sig = torch.sigmoid(self.sigma.unsqueeze(0) * (v - self.mu.unsqueeze(0)))  # (B, H, H)
            w_activation = self.W.unsqueeze(0) * rec_sig  # (B, H, H)
            rev_activation = w_activation * self.erev.unsqueeze(0)  # (B, H, H)

            w_numerator = rev_activation.sum(dim=2) + w_numerator_sensory  # (B, H)
            w_denominator = w_activation.sum(dim=2) + w_denominator_sensory  # (B, H)

            numerator = self.cm_t.unsqueeze(0) * v_pre + self.gleak.unsqueeze(0) * self.vleak.unsqueeze(0) + w_numerator
            denominator = self.cm_t.unsqueeze(0) + self.gleak.unsqueeze(0) + w_denominator

            v_pre = numerator / denominator

        return v_pre

    # ODE的导数计算
    def _f_prime(self, inputs, state):
        v_pre = state
        inp = inputs.unsqueeze(-1)
        sensory_sig = torch.sigmoid(self.sensory_sigma.unsqueeze(0) * (inp - self.sensory_mu.unsqueeze(0)))
        sensory_w_activation = self.sensory_W.unsqueeze(0) * sensory_sig
        w_reduced_sensory = sensory_w_activation.sum(dim=1)

        v = v_pre.unsqueeze(-1)
        w_activation = self.W.unsqueeze(0) * torch.sigmoid(self.sigma.unsqueeze(0) * (v - self.mu.unsqueeze(0)))
        w_reduced_synapse = w_activation.sum(dim=2)

        sensory_in = (self.sensory_erev.unsqueeze(0) * sensory_w_activation).sum(dim=1)
        synapse_in = (self.erev.unsqueeze(0) * w_activation).sum(dim=2)

        sum_in = sensory_in + synapse_in - v_pre * (w_reduced_synapse + w_reduced_sensory)

        f_prime = (self.gleak.unsqueeze(0) * (self.vleak.unsqueeze(0) - v_pre) + sum_in) / self.cm_t.unsqueeze(0)
        return f_prime

    # RK4 ODE求解步骤
    def _ode_step_rk4(self, inputs, state):
        h = 0.1
        v = state
        for _ in range(self.ode_unfolds):
            k1 = h * self._f_prime(inputs, v)
            k2 = h * self._f_prime(inputs, v + 0.5 * k1)
            k3 = h * self._f_prime(inputs, v + 0.5 * k2)
            k4 = h * self._f_prime(inputs, v + k3)
            v = v + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return v

    # 显式ODE求解步骤
    def _ode_step_explicit(self, inputs, state):
        v_pre = state
        inp = inputs.unsqueeze(-1)
        sensory_sig = torch.sigmoid(self.sensory_sigma.unsqueeze(0) * (inp - self.sensory_mu.unsqueeze(0)))
        sensory_w_activation = self.sensory_W.unsqueeze(0) * sensory_sig
        w_reduced_sensory = sensory_w_activation.sum(dim=1)

        for _ in range(self.ode_unfolds):
            v = v_pre.unsqueeze(-1)
            w_activation = self.W.unsqueeze(0) * torch.sigmoid(self.sigma.unsqueeze(0) * (v - self.mu.unsqueeze(0)))
            w_reduced_synapse = w_activation.sum(dim=2)

            sensory_in = (self.sensory_erev.unsqueeze(0) * sensory_w_activation).sum(dim=1)
            synapse_in = (self.erev.unsqueeze(0) * w_activation).sum(dim=2)

            sum_in = sensory_in + synapse_in - v_pre * (w_reduced_synapse + w_reduced_sensory)
            f_prime = (self.gleak.unsqueeze(0) * (self.vleak.unsqueeze(0) - v_pre) + sum_in) / self.cm_t.unsqueeze(0)

            v_pre = v_pre + 0.1 * f_prime

        return v_pre

# LTCRNN包装器
class LTCRNN(nn.Module):
    """通过时间序列运行LTCCell的包装器。返回hs (T, B, H)。"""
    def __init__(self, input_size, hidden_size, solver=ODESolver.SemiImplicit, ode_unfolds=6):
        super().__init__()
        self.cell = LTCCell(input_size=input_size, hidden_size=hidden_size, solver=solver, ode_unfolds=ode_unfolds)

    def forward(self, seq):
        # seq: (T, B, input_size)
        T, B, _ = seq.shape
        h = torch.zeros(B, self.cell.hidden_size, device=seq.device)
        hs = []
        for t in range(T):
            h = self.cell(seq[t], h)
            hs.append(h.unsqueeze(0))
        return torch.cat(hs, dim=0)  # (T, B, H)


# ----------------------------
# 模型：卷积神经网络（CNN）→ 长时序循环神经网络（LTCRNN）建模时序动态 → Multi-Head Attention（MHA）学习全局依赖 → 分类器 → Stratified K-Fold（5折）训练
# ----------------------------

# CNN特征提取
class CNNFeature(nn.Module):
    def __init__(self, in_channels=10, cnn_feat=CNN_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),  # Conv1d 提取局部时空模式
            nn.BatchNorm1d(32),                     # 批归一化
            nn.ReLU(inplace=True),                  # 激活函数
            nn.Conv1d(32, cnn_feat, kernel_size=3, padding=1),     # 第二层卷积
            nn.BatchNorm1d(cnn_feat),               # 批归一化
            nn.ReLU(inplace=True)                   # 激活函数
        )  

    def forward(self, x):
        # x: (Batch, F, T)
        return self.net(x)  # (Batch, C, T)

# 完整模型：CNN + LTC + MHA
class CNN_LTC_MHA(nn.Module):
    def __init__(self, in_channels=10, cnn_feat=CNN_FEAT, ltc_hidden=LTC_HIDDEN, num_classes=NUM_CLASSES, mha_heads=MHA_HEADS, solver=ODESolver.SemiImplicit):
        super().__init__()

        # 1. CNN特征提取
        self.cnn = CNNFeature(in_channels=in_channels, cnn_feat=cnn_feat)

        # 2. LTC RNN建模时序动态
        self.ltc = LTCRNN(input_size=cnn_feat, hidden_size=ltc_hidden, solver=solver)
        
        # 3. 多头注意力（MHA）
        self.mha = nn.MultiheadAttention(embed_dim=ltc_hidden, num_heads=mha_heads, batch_first=False)
        self.classifier = nn.Linear(ltc_hidden, num_classes)

    def forward(self, x):
        # x: (B, F, T)
        feat = self.cnn(x)  # (B, C, T)
        # 转换为 (T, B, C)
        seq = feat.permute(2, 0, 1).contiguous()
        hs = self.ltc(seq)  # (T, B, H)
        # 应用多头注意力（时间维度的自我注意力）
        # 注意：当 batch_first=False 时，nn.MultiheadAttention 期望输入格式为 (L, N, E)
        attn_out, attn_weights = self.mha(hs, hs, hs)  # attn_out: (T, B, H)
        # 聚合：时间维度上的注意力加权平均值
        # 我们可以在时间维度上对attention_out进行求平均
        context = attn_out.mean(dim=0)  # (B, H)
        logits = self.classifier(context)  # (B, num_classes)
        return logits, attn_weights  # attn_weights: (B, T, T)


# ----------------------------
# 训练函数：K折交叉验证训练CNN + LTC + MHA模型
# ----------------------------
def run_training_mha(data_root,
                     k=K_FOLDS,
                     fixed_len=FIXED_LEN,
                     cnn_feat=CNN_FEAT,
                     ltc_hidden=LTC_HIDDEN,
                     epochs=EPOCHS,
                     batch_size=BATCH_SIZE,
                     lr=LR,
                     patience=PATIENCE,
                     seed=SEED,
                     output_dir=OUTPUT_DIR):
    files, labels = collect_files_labels(data_root)
    print(f"Found {len(files)} samples.")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold = 0
    fold_accs = []
    for train_idx, val_idx in skf.split(files, labels):
        fold += 1
        print("\n" + "="*30)
        print(f"Fold {fold}/{k}  (train {len(train_idx)}  val {len(val_idx)})")
        train_files = files[train_idx]; train_labels = labels[train_idx]
        val_files = files[val_idx]; val_labels = labels[val_idx]

        # build per-fold scaler from train set (concatenate rows)
        rows = []
        for p in train_files:
            try:
                df = pd.read_excel(p, engine='openpyxl')
            except Exception:
                df = pd.read_excel(p)
            arr = df.values.astype(np.float32)[:fixed_len]
            rows.append(arr)
        rows = np.vstack(rows)
        scaler = StandardScaler().fit(rows)

        def tr_transform(arr):
            T, F = arr.shape
            scaled = scaler.transform(arr.reshape(-1, F)).reshape(T, F)
            return scaled.astype(np.float32)

        def va_transform(arr):
            T, F = arr.shape
            scaled = scaler.transform(arr.reshape(-1, F)).reshape(T, F)
            return scaled.astype(np.float32)

        train_ds = GasDataset(train_files, train_labels, fixed_len=fixed_len, transform=tr_transform)
        val_ds = GasDataset(val_files, val_labels, fixed_len=fixed_len, transform=va_transform)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fixed, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fixed, num_workers=0)

        # model
        in_channels = train_ds[0][0].shape[1]  # number of sensors (should be 10)
        model = CNN_LTC_MHA(in_channels=in_channels, cnn_feat=cnn_feat, ltc_hidden=ltc_hidden).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state = None
        wait = 0

        for ep in range(1, epochs+1):
            model.train()
            total_loss = 0.0
            t0 = time.time()
            for xb, yb, paths in train_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                optimizer.zero_grad()
                logits, _ = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * yb.size(0)
                # clamp LTC parameters if needed
                try:
                    # The LTCCell has clamp_parameters method inside; apply if present
                    if hasattr(model.ltc.cell, "clamp_parameters"):
                        model.ltc.cell.clamp_parameters()
                except Exception:
                    pass
            avg_train_loss = total_loss / len(train_ds)

            # validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for xb, yb, paths in val_loader:
                    xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                    logits, attn = model(xb)
                    loss = loss_fn(logits, yb)
                    val_loss += loss.item() * yb.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
                    all_preds.extend(preds.cpu().tolist())
                    all_trues.extend(yb.cpu().tolist())
            avg_val_loss = val_loss / len(val_ds)
            val_acc = correct / total

            print(f"Fold{fold} Ep{ep}/{epochs} TrainLoss={avg_train_loss:.4f} ValLoss={avg_val_loss:.4f} ValAcc={val_acc:.4f}")

            # scheduler step
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    'model_state_dict': model.state_dict(),
                    'scaler': scaler,
                    'cnn_feat': cnn_feat,
                    'ltc_hidden': ltc_hidden,
                    'mha_heads': MHA_HEADS
                }
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("[INFO] Early stopping")
                    break

        # Save best for this fold
        if best_state is not None:
            fname = os.path.join(output_dir, f"ltc_mha_best_fold{fold}.pt")
            torch.save(best_state, fname)
            print(f"[SAVED] Fold {fold} best model -> {fname}")

        # per-fold final metrics
        fold_accs.append(best_val_acc)
        print(f"[FOLD {fold}] best_val_acc = {best_val_acc:.4f}")

    print("\nAll folds best accs:", fold_accs)
    print("Mean acc:", float(np.mean(fold_accs)))
    return fold_accs


# ----------------------------
# 快速烟雾测试（示例）
# ----------------------------
def smoke_check():
    print("DATA_ROOT:", DATA_ROOT)
    print("EXAMPLE_FILE exists:", os.path.exists(EXAMPLE_FILE))
    if os.path.exists(EXAMPLE_FILE):
        try:
            df = pd.read_excel(EXAMPLE_FILE, engine='openpyxl')
        except Exception:
            df = pd.read_excel(EXAMPLE_FILE)
        arr = df.values.astype(np.float32)
        print("Example shape (time, sensors):", arr.shape)
        print("First 3 rows:\n", arr[:3])
    else:
        print("Example file missing; ensure EXAMPLE_FILE path correct.")


# ----------------------------
# Run 
# ----------------------------
if __name__ == "__main__":
    # smoke_check()
    # To actually train K-fold, uncomment below:
    fold_accs = run_training_mha(DATA_ROOT, k=K_FOLDS, fixed_len=FIXED_LEN,
                                  cnn_feat=CNN_FEAT, ltc_hidden=LTC_HIDDEN,
                                  epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                                  patience=PATIENCE, seed=SEED, output_dir=OUTPUT_DIR)

#All folds best accs: [0.9928571428571429, 0.9785714285714285, 0.95, 0.95, 0.9571428571428572]

