# Literature Review: 基于概率深度学习的剩余使用寿命预测与不确定性量化

## 1. 引言

剩余使用寿命（Remaining Useful Life, RUL）预测是预测性维护（Predictive Maintenance, PdM）和故障预测与健康管理（Prognostics and Health Management, PHM）的核心任务。准确的 RUL 预测使运维人员能够在设备失效前合理安排维护，减少非计划停机和过度维护带来的经济损失（Lei et al., 2018）。近年来，深度学习方法在 RUL 预测中取得了显著进展，但绝大多数方法仅提供点估计，缺乏对预测不确定性的量化——而这在安全关键场景中是不可或缺的。

## 2. 数据驱动的 RUL 预测方法演进

### 2.1 传统机器学习方法

早期数据驱动方法主要依赖手工特征工程与传统回归模型。Mosallam et al. (2016) 使用支持向量回归（SVR）进行 RUL 预测；Ramasso & Saxena (2014) 提出了基于隐马尔可夫模型的方法。这些方法依赖领域专家的特征设计，泛化能力有限。

### 2.2 深度学习方法

深度学习通过自动特征提取显著提升了 RUL 预测性能：

- **CNN 方法**：Sateesh Babu et al. (2016) 首次将 CNN 应用于 RUL 预测，利用卷积层提取局部时序模式。Li et al. (2018) 提出的多尺度 CNN 在 C-MAPSS 上取得了优异性能。

- **RNN/LSTM 方法**：Zheng et al. (2017) 和 Wu et al. (2018) 将 LSTM 引入 RUL 预测，利用其门控机制建模长期时序依赖。双向 LSTM（Bi-LSTM）进一步增强了对序列的全局理解（Wang et al., 2019）。LSTM 已成为 RUL 预测中最广泛使用的深度模型之一。

- **注意力与 Transformer 方法**：Li et al. (2019) 引入注意力机制加权关键时间步；Mo et al. (2021) 将 Transformer 直接用于 RUL 预测，利用自注意力机制捕获全局时序关系，避免了 RNN 的梯度消失问题。

- **混合方法**：CNN-LSTM 混合架构结合了 CNN 的特征提取能力与 LSTM 的时序建模能力（Al-Dulaimi et al., 2019），在多个基准上表现优异。

### 2.3 C-MAPSS 基准数据集

NASA C-MAPSS（Saxena et al., 2008）是 RUL 预测领域最广泛使用的基准，包含 4 个子集（FD001-FD004），模拟不同运行条件和故障模式下的涡轮风扇发动机退化过程。其中 FD001 为单一运行条件、单一故障模式，是入门基准；FD003/FD004 引入多故障模式，更具挑战性。分段线性 RUL 标注方案（R_early 截断）由 Heimes (2008) 提出，已成为标准做法。

## 3. 预测不确定性量化

### 3.1 不确定性的分类

Kiureghian & Ditlevsen (2009) 将预测不确定性分为两类：
- **偶然不确定性（Aleatoric）**：数据固有噪声，不可通过增加数据消除
- **认知不确定性（Epistemic）**：模型知识不足导致，可通过更多数据或更好模型缓解

### 3.2 深度学习中的不确定性量化方法

- **贝叶斯神经网络（BNN）**：在权重上施加先验分布，通过变分推断近似后验（Blundell et al., 2015）。理论优雅但计算昂贵，参数量翻倍。

- **MC Dropout**：Gal & Ghahramani (2016) 证明在测试时启用 Dropout 等价于近似贝叶斯推断。通过多次随机前向传播获取预测分布，以极低的实现成本获得认知不确定性估计。这是目前最广泛使用的方法之一。

- **深度集成（Deep Ensemble）**：Lakshminarayanan et al. (2017) 提出训练多个独立模型并聚合预测，在许多任务中不确定性校准优于 MC Dropout，但训练成本随集成数线性增长。

- **概率输出头**：Nix & Weigend (1994) 的开创性工作提出让网络同时输出均值和方差，通过高斯负对数似然（Gaussian NLL）损失训练，直接捕获偶然不确定性。

### 3.3 RUL 预测中的不确定性量化

将不确定性量化引入 RUL 预测是近年的重要研究方向。Kraus & Feuerriegel (2019) 首次在 LSTM-based RUL 预测中使用 MC Dropout 估计认知不确定性。Biggio et al. (2021) 提出将概率输出（学习异方差噪声）与 MC Dropout 结合，实现偶然-认知不确定性的联合分解。Peng et al. (2020) 使用贝叶斯 LSTM 进行 RUL 的区间预测。Li et al. (2020) 探索了深度集成在 RUL 预测中的应用。

然而，现有工作在以下方面仍有不足：不确定性校准质量的系统评估较少；不确定性感知的维护决策框架尚未成熟；多工况/多故障模式下不确定性行为的分析有限。

## 4. 本项目的定位

本项目实现了概率 LSTM + MC Dropout 的联合框架，在 C-MAPSS FD001 上同时捕获偶然和认知不确定性。其贡献在于：（1）通过高斯 NLL 损失学习异方差偶然不确定性；（2）通过 MC Dropout 估计认知不确定性；（3）提供了系统的不确定性质量评估（PICP、校准曲线、稀疏化图等）。实验表明，概率模型在不牺牲点预测精度（RMSE 14.84 vs 14.96）的前提下，实现了 93% 的 95% 置信区间覆盖率。

## 5. 待探索的 Research Questions

基于上述文献和本项目的实验结果，提出以下有价值的研究方向：

### RQ1: 如何在多工况、多故障模式下维持不确定性估计的校准质量？

本项目仅在 FD001（单工况/单故障）上验证。FD002-FD004 引入了 6 种运行条件和 2 种故障模式，数据分布更复杂。**分布外（OOD）样本的不确定性行为**是否仍可靠？模型能否为未见过的运行条件给出合理的高认知不确定性？这直接关系到实际部署的可靠性。

### RQ2: 不确定性估计能否改善维护决策的经济效益？

现有研究多关注预测精度和不确定性校准的统计指标，但缺乏将不确定性**直接嵌入决策框架**的工作。例如：基于预测 RUL 分布（而非点估计）设计风险感知的维护调度策略，最小化期望总成本（非计划故障成本 + 预防性更换成本）。NASA Score 的非对称惩罚已暗示了这一需求。

### RQ3: 偶然-认知不确定性分解对模型改进有何指导意义？

本项目结果显示偶然不确定性（std=12.33）远大于认知不确定性（std=4.38）。这意味着增加模型容量或训练数据的边际收益有限，而**改善输入数据质量**（如传感器融合、降噪）可能更有效。能否利用这种分解**自适应地引导数据采集或特征工程**？

### RQ4: Transformer 架构下的概率预测与不确定性量化效果如何？

LSTM 的序列建模能力受限于梯度传播和固定窗口长度。将概率输出头和 MC Dropout 框架迁移到 **Transformer / Temporal Fusion Transformer** 上，能否在长序列和多变量依赖建模上获得更好的不确定性估计？注意力权重的可解释性能否辅助不确定性的来源分析？

### RQ5: 轻量级不确定性量化方法能否满足边缘部署需求？

MC Dropout 需要 T 次前向传播（本项目 T=100），这在资源受限的边缘设备上不可行。**单次前向传播**的不确定性估计方法（如 Spectral-normalized Neural Gaussian Process、Evidential Deep Learning）能否在保持校准质量的同时大幅降低推理成本？

## 参考文献

| 领域 | 关键文献 |
|------|---------|
| RUL 综述 | Lei et al. (2018), *Mech. Syst. Signal Process.* |
| C-MAPSS 数据集 | Saxena et al. (2008), *PHM Conference* |
| RUL 标注方案 | Heimes (2008), *PHM Conference* |
| LSTM for RUL | Zheng et al. (2017); Wu et al. (2018) |
| Bi-LSTM for RUL | Wang et al. (2019) |
| CNN for RUL | Sateesh Babu et al. (2016); Li et al. (2018) |
| 注意力机制 for RUL | Li et al. (2019) |
| Transformer for RUL | Mo et al. (2021) |
| CNN-LSTM 混合 | Al-Dulaimi et al. (2019) |
| 不确定性分类 | Kiureghian & Ditlevsen (2009) |
| BNN | Blundell et al. (2015), *ICML* |
| MC Dropout 理论 | Gal & Ghahramani (2016), *ICML* |
| 深度集成 | Lakshminarayanan et al. (2017), *NeurIPS* |
| 概率输出 | Nix & Weigend (1994), *ICNN* |
| RUL + MC Dropout | Kraus & Feuerriegel (2019) |
| RUL + 概率 + MC Dropout | Biggio et al. (2021) |
| 贝叶斯 LSTM for RUL | Peng et al. (2020) |
| 深度集成 for RUL | Li et al. (2020) |
| SVR for RUL | Mosallam et al. (2016) |
| HMM for RUL | Ramasso & Saxena (2014) |
