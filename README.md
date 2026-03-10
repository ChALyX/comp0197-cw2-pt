[English](#english) | [中文](#中文)

---

<a id="中文"></a>

# 概率 RUL 预测与不确定性量化

## 项目简介

本项目实现了一个**概率 LSTM** 模型，用于预测涡轮风扇发动机的**剩余使用寿命（RUL）**，并内置**不确定性量化**功能。该项目为 UCL COMP0197 Applied Deep Learning 课程作业。

模型输出高斯分布（均值和标准差）而非单一点估计，并使用 **MC Dropout** 将预测不确定性分解为：

- **偶然不确定性（Aleatoric）** — 数据固有噪声，通过学习的方差捕获
- **认知不确定性（Epistemic）** — 模型不确定性，通过 MC Dropout 采样捕获

## 数据集

**NASA C-MAPSS（商用模块化航空推进系统仿真）— FD001 子集**

| 属性 | 值 |
|------|-----|
| 训练轨迹 | 100 台发动机（完整退化至故障） |
| 测试轨迹 | 100 台发动机（在故障前截断） |
| 运行条件 | 1 种（海平面） |
| 故障模式 | 1 种（HPC 退化） |
| 每时间步传感器数 | 21 个传感器 + 3 个运行设置 |
| 选用特征数 | 14 个（通过标准差 < 0.01 移除常数传感器） |

每行包含 26 列空格分隔数据：`unit_id`、`cycle`、3 个运行设置和 21 个传感器读数。RUL 标签采用分段线性方法构造，上限截断为 125 个周期。

运行 `train.py` 时数据集会从 NASA 开放数据门户自动下载。

## 模型架构

### 概率 LSTM（主模型）

- 2 层 LSTM 编码器，层间带 Dropout
- 两个输出头：**mu**（均值）和 **sigma**（标准差），通过 `exp(log_sigma) + 1e-6` 保证正值
- 使用**高斯负对数似然损失（Gaussian NLL Loss）** 训练
- 测试时进行 **T=100 次 MC Dropout 前向传播**以估计认知不确定性

### 确定性 LSTM（基线模型）

- 相同 LSTM 架构，单输出头
- 使用 **MSE 损失**训练
- 用于消融实验对比

### 超参数

| 参数 | 值 |
|------|-----|
| 序列长度 | 30 |
| 隐藏层维度 | 128 |
| LSTM 层数 | 2 |
| Dropout 比率 | 0.25 |
| 学习率 | 0.001（Adam） |
| 权重衰减 | 1e-4 |
| 批大小 | 32 |
| R_early（RUL 上限） | 125 |
| 学习率调度 | ReduceLROnPlateau（factor=0.5, patience=5） |
| 早停耐心值 | 20 |
| MC Dropout 采样次数 | 100 |

## 实验结果

### 点预测指标

| 模型 | RMSE | MAE | R² | NASA 评分 |
|------|------|-----|-----|----------|
| **概率 LSTM** | **14.84** | **11.08** | **0.872** | **351.07** |
| 确定性 LSTM | 14.96 | 11.12 | 0.870 | 424.06 |

### 不确定性质量指标（概率 LSTM）

| 指标 | 值 |
|------|-----|
| PICP（95% 置信区间覆盖率） | 0.93 |
| MPIW（95% 置信区间宽度） | 51.64 |
| NLL（负对数似然） | 3.02 |
| 平均偶然不确定性标准差 | 12.33 |
| 平均认知不确定性标准差 | 4.38 |
| 平均总不确定性标准差 | 13.17 |

概率模型在获得与确定性模型相当甚至更优的点预测性能的同时，还提供了校准良好的不确定性估计（95% 置信区间实际覆盖率为 93%）。

## 使用方法

```bash
# 环境配置
micromamba activate comp0197-pt
pip install matplotlib pandas scikit-learn

# 训练两个模型
python train.py

# 评估并生成图表
python test.py
```

## 项目结构

```
├── train.py              # 训练流程（下载、预处理、训练、保存）
├── test.py               # 评估流程（MC Dropout 推理、指标计算、绘图）
├── models/
│   ├── probabilistic_lstm.py   # 概率 LSTM 模型
│   └── deterministic_lstm.py   # 确定性基线模型
├── utils/
│   ├── data_loader.py    # 数据下载、预处理、Dataset/DataLoader
│   ├── metrics.py        # RMSE、MAE、R²、NASA 评分、PICP、MPIW、NLL、校准曲线
│   ├── visualization.py  # 所有 matplotlib 图表
│   └── helpers.py        # EarlyStopping、随机种子设置
├── saved_models/         # 模型检查点（.pth）
└── results/
    ├── figures/          # 生成的 PNG 图表
    └── metrics.json      # 量化结果
```

---

<a id="english"></a>

# Probabilistic RUL Prediction with Uncertainty Quantification

## Project Overview

This project implements a **Probabilistic LSTM** model for predicting the **Remaining Useful Life (RUL)** of turbofan engines, with built-in **uncertainty quantification**. It is developed as part of the UCL COMP0197 Applied Deep Learning coursework.

The model outputs a Gaussian distribution (mean and standard deviation) rather than a single point estimate, and uses **MC Dropout** to decompose predictive uncertainty into:

- **Aleatoric uncertainty** — inherent data noise, captured by the learned variance
- **Epistemic uncertainty** — model uncertainty, captured by MC Dropout sampling

## Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) — FD001 subset**

| Property | Value |
|----------|-------|
| Training trajectories | 100 engines (run-to-failure) |
| Test trajectories | 100 engines (truncated before failure) |
| Operating conditions | 1 (sea level) |
| Fault modes | 1 (HPC degradation) |
| Sensors per timestep | 21 sensors + 3 operational settings |
| Selected features | 14 (constant sensors removed by std < 0.01) |

Each row contains 26 space-separated columns: `unit_id`, `cycle`, 3 operational settings, and 21 sensor readings. RUL labels are constructed using a piece-wise linear scheme with an upper bound of 125 cycles.

The dataset is automatically downloaded from NASA Open Data Portal when running `train.py`.

## Model Architecture

### Probabilistic LSTM (Main Model)

- 2-layer LSTM encoder with dropout between layers
- Two output heads: **mu** (mean) and **sigma** (std) via `exp(log_sigma) + 1e-6`
- Trained with **Gaussian Negative Log-Likelihood Loss**
- At test time, runs **T=100 MC Dropout forward passes** to estimate epistemic uncertainty

### Deterministic LSTM (Baseline)

- Same LSTM architecture, single output head
- Trained with **MSE Loss**
- Used for ablation comparison

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence length | 30 |
| Hidden dimension | 128 |
| LSTM layers | 2 |
| Dropout | 0.25 |
| Learning rate | 0.001 (Adam) |
| Weight decay | 1e-4 |
| Batch size | 32 |
| R_early (RUL cap) | 125 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping patience | 20 |
| MC Dropout samples | 100 |

## Results

### Point Prediction Metrics

| Model | RMSE | MAE | R² | NASA Score |
|-------|------|-----|-----|------------|
| **Probabilistic LSTM** | **14.84** | **11.08** | **0.872** | **351.07** |
| Deterministic LSTM | 14.96 | 11.12 | 0.870 | 424.06 |

### Uncertainty Quality Metrics (Probabilistic LSTM)

| Metric | Value |
|--------|-------|
| PICP (95% CI) | 0.93 |
| MPIW (95% CI) | 51.64 |
| NLL | 3.02 |
| Mean Aleatoric Std | 12.33 |
| Mean Epistemic Std | 4.38 |
| Mean Total Std | 13.17 |

The probabilistic model achieves competitive point prediction performance while providing calibrated uncertainty estimates (93% coverage for 95% confidence intervals).

## Usage

```bash
# Environment setup
micromamba activate comp0197-pt
pip install matplotlib pandas scikit-learn

# Train both models
python train.py

# Evaluate and generate figures
python test.py
```

## Project Structure

```
├── train.py              # Training pipeline (download, preprocess, train, save)
├── test.py               # Evaluation pipeline (MC Dropout inference, metrics, plots)
├── models/
│   ├── probabilistic_lstm.py
│   └── deterministic_lstm.py
├── utils/
│   ├── data_loader.py    # Data download, preprocessing, Dataset/DataLoader
│   ├── metrics.py        # RMSE, MAE, R², NASA Score, PICP, MPIW, NLL, calibration
│   ├── visualization.py  # All matplotlib figures
│   └── helpers.py        # EarlyStopping, seed setup
├── saved_models/         # Model checkpoints (.pth)
└── results/
    ├── figures/          # Generated PNG plots
    └── metrics.json      # Quantitative results
```
