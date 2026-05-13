# Pollutants-FCNN 污染物毒性预测系统
**基于真实 Tox21 数据的多任务深度学习框架**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/Python->=3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)
**暨南大学环境与气候学院 张耀霖 Email：zhangyaolin@stu2022.jnu.edu.cn**

**Pollutants-FCNN** 是一个基于多任务学习（Multi-Task Learning, MTL）和前馈神经网络（FCNN）的计算毒理学框架。该项目旨在利用深度学习技术对环境污染物进行高效、低成本的毒性预测，替代或辅助传统高成本、耗时的动物实验。

本项目复现了论文《Application of the Pollutants-FCNN Framework: A Multi-Task Neural Network Approach for Contaminant Toxicity Prediction Based on Tox21 Data》的核心算法，并提供了一个交互式桌面应用程序（GUI），支持模型训练、评估及单样本预测。

---

## 📚 核心特性

### 🧬 学术背景与模型架构
本项目基于 **Tox21** 基准数据集开发，该数据集包含 801 维化学特征。模型将 12 种原始毒性终点（Endpoints）聚合成 4 大类主要毒性，以解决特定终点数据稀缺的问题。

- **论文来源**: *Climate Sustainability & Global Systems* (2026)
- **核心创新**: 结合了 **多任务学习 (MTL)** 与 **多头注意力机制 (Multi-Head Attention)**，有效捕捉特征间的交互作用。
- **性能指标**: 在测试集上达到了 **0.789** 的宏平均 AUC (Macro-Average AUC)，优于传统的单任务学习模型。

### 🛠️ 功能亮点
- **数据自动处理**: 自动下载 Tox21 数据集，处理缺失值（NaN），并自动将 12 个原始指标聚合成 4 类毒性。
- **交互式 GUI**: 使用 PyQt5 构建的桌面端界面，无需命令行即可完成模型训练与预测。
- **多毒性预测**: 同时预测以下 4 种环境毒性类型：
  - **生物毒性 (Biological Toxicity)**
  - **细胞毒性 (Cell Toxicity)**
  - **神经毒性 (Neurotoxicity)**
  - **遗传毒性 (Genotoxicity)**
- **模型持久化**: 支持模型的保存（`.pth`）与加载，方便后续使用。

---

## 🚀 快速开始

### 1. 环境依赖
请确保您的环境已安装 Python 3.8+ 及以下库：

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn pyqt5 requests tqdm
###2. 项目结构
克隆项目后，建议创建以下目录结构：
bash

编辑



mkdir -p data/tox21 models
###3. 启动应用
直接运行主程序文件：
bash

编辑



python fcnnpredict.py
💻 使用指南
🧠 模型训练
点击 "下载Tox21数据集" (可选，程序在训练时也会自动检查并下载)。
点击 "开始训练模型"。
系统将自动执行：数据预处理 -> 特征标准化 -> 模型训练（含验证集监控）-> 保存模型。
训练过程中会弹出可视化窗口，展示 Loss 和 AUC 曲线。
🔮 毒性预测
加载模型: 点击 "加载已训练模型" 选择 .pth 文件。
输入特征:
在文本框中直接输入 801 个特征值（空格或换行分隔）。
或点击 "加载特征文件" 上传包含特征值的文本/CSV文件。
点击 "开始预测"。
结果解读: 输出包含每种毒性的概率值（0-1）、预测结果（阳性/阴性）及置信度。
📊 毒性分类详情
模型将化学物质的毒性划分为以下四大类，具体映射关系如下：
表格
毒性类别	包含的原始指标 (Endpoints)
细胞毒性	NR-AR, NR-AR-LBD, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-HSE, SR-MMP
遗传毒性	NR-AhR, SR-ATAD纯, SR-p53
神经毒性	NR-Aromatase
生物毒性	SR-ARE
📂 代码模块说明
fcnnpredict.py 文件包含了完整的端到端流程：
数据处理 (Tox21DataProcessor):
实现了 Tox21 数据的网络获取、缺失值掩码处理及毒性类型聚合逻辑。
模型架构 (PollutantsFCNN):
网络结构：Input (801D) -> Projection (1024D) -> FCNN Layers (512->256D) -> Multi-Head Attention -> Task-Specific Heads。
使用了 BatchNorm 和 Dropout 防止过拟合。
训练器 (ToxicityPredictor):
使用加权二元交叉熵（Weighted BCE）处理类别不平衡。
集成了早停法（Early Stopping）和学习率调度。
可视化与 GUI:
提供了训练历史绘图功能。
全功能 PyQt5 图形用户界面。
🎓 引用与致谢
如果您在研究中使用了本项目或参考了相关代码，请引用原论文：
Zhang, Y., Li, M., & Zou, Y. (2026). Application of the Pollutants-FCNN Framework: A Multi-Task Neural Network Approach for Contaminant Toxicity Prediction Based on Tox21 Data. Climate Sustainability & Global Systems.
数据来源:
Tox21 Challenge Dataset (https://bioinf.jku.at/research/DeepTox/)
开发环境:
Python 3.9
PyTorch 1.12
Windows/Linux/macOS
