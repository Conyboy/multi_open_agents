# Multi-Agent Time Series Forecasting System

## 项目简介

本项目是一个基于多智能体（Multi-Agent）架构的时间序列预测系统，专为风电功率预测任务设计。项目结合了传统机器学习方法（LightGBM、XGBoost）与大语言模型（LLM）智能体，通过自动化特征工程和超参数调优来提升预测准确性。

## 核心特性

- **多智能体架构**：将复杂任务分解给专门的AI智能体处理
- **自动化机器学习**：实现自动特征工程、参数调优和模型选择
- **LLM辅助决策**：利用大语言模型在特征选择和参数调优方面提供专业建议
- **可扩展设计**：模块化架构便于添加新的特征或模型

## 技术栈

- Python 3.x
- LightGBM/XGBoost 机器学习模型
- Scikit-learn 数据预处理和评估
- Pandas/Numpy 数据处理
- Matplotlib/Seaborn 数据可视化
- OpenAgents 多智能体框架

## 项目结构

```
.
├── agents/                     # 智能体实现目录
│   ├── agent_a_load_analyze.py # 数据加载与分析智能体
│   ├── agent_b_feature_select.py# 特征选择智能体
│   ├── agent_c_model_tune.py   # 模型调参智能体
│   ├── agent_d_train_eval.py   # 训练评估智能体
│   └── agent_e_analyze_result.py# 结果分析智能体
├── sample_data/                # 示例数据目录
│   ├── A_train_data.csv        # 训练数据
│   ├── A_test_data.csv         # 测试数据
│   └── A_train_info.csv        # 训练信息
├── default_params/             # 默认参数配置目录
│   ├── feature_library.json    # 特征库定义
│   └── model_param.json        # 模型参数模板
├── experiments/                # 实验结果目录
├── requirements.txt            # 项目依赖
└── launch_network.sh           # 项目启动脚本
```

## 演示视频

*模型自动优化训练多智能体录屏((模型训练完成在2分0秒).mp4*

<iframe width="600" height="400" src="https://www.bilibili.com/video/BV1ENqgBnEAj/?share_source=copy_web&vd_source=b25bcd86e5176cbdebc240259a9aeba0" frameborder="0" allowfullscreen></iframe>


## 工作原理

### 五个核心智能体

1. **Agent A - 数据分析智能体**
   - 负责加载和初步分析时间序列数据
   - 提取数据基本信息如形状、时间范围、列相关性等

2. **Agent B - 特征选择智能体**
   - 基于特征库和LLM建议选择合适特征
   - 使用大语言模型决定哪些特征对当前预测任务最有价值

3. **Agent C - 模型调参智能体**
   - 使用LLM为LightGBM和XGBoost模型推荐最优超参数
   - 决定模型集成权重

4. **Agent D - 训练评估智能体**
   - 执行实际的模型训练和评估过程
   - 使用选定特征和参数训练LightGBM和XGBoost模型
   - 进行模型集成并生成评估指标

5. **Agent E - 结果分析智能体**
   - 分析每轮迭代的结果
   - 绘制性能趋势图
   - 决定是否继续下一轮迭代或提前停止

### 特征工程技术

- 时间特征提取（年、月、日、小时等）
- 特征交叉（风速、风向等气象数据的组合）
- 风向周期化处理
- 数据标准化/归一化
- 滞后特征构建
- 缺失值填充

## 快速开始

### 环境配置

```bash
conda create -n openagents python=3.12 -y
conda activate openagents
pip install -r requirements.txt
```

### 运行项目

1. 启动网络服务：
```bash
openagents init ./network
openagents network start ./my_ml_network > ./logs/network.log  &
```

2. 依次启动各智能体：
```bash
python agents/agent_a_load_analyze.py > ./logs/agent_a.log &
python agents/agent_b_feature_select.py > ./logs/agent_a.log  &
python agents/agent_c_model_tune.py > ./logs/agent_a.log  &
python agents/agent_d_train_eval.py > ./logs/agent_a.log  &
python agents/agent_e_analyze_result.py > ./logs/agent_a.log  &
```

3. 触发数据处理流程：
通过向频道里面发送包含"数据加载"关键字的消息启动整个流程。

## 使用场景

本项目特别适用于：
- 风电功率预测
- 其他需要高精度时间序列预测的场景
- 受多种因素影响的复杂预测任务

## 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情
