# H2-LLM Mapping Strategy Implementation

本文档介绍了 H2-LLM 映射策略的实现，该策略基于 ISCA 2025 论文：
**"H2-LLM: Hardware-Dataflow Co-Exploration for Heterogeneous Hybrid-Bonding-based Low-Batch LLM Inference"**

## 概述

H2-LLM 是一个针对边缘端低批次 LLM 推理的异构混合键合加速器。本实现包含了论文中提出的核心映射算法。

## 核心算法

### 1. 跨通道算子分区（Inter-Channel Operator Partition）

基于论文第 4.2 节的解析模型，该算法通过最小化数据传输开销来确定最优的分片因子。

#### 问题建模

对于 GEMM 算子，输入张量形状为 `(M, K)`，权重张量形状为 `(K, N)`，需要在 C 个 NMP 通道上进行分区。

**优化目标：** 最小化总传输开销

```
min_{T_K, T_N} s × M × (K/(T_K × B_s) + N/(T_N × B_l))
```

**约束条件：**
```
T_K × T_N = C
```

其中：
- `s`: 元素大小（字节）
- `M, K, N`: 矩阵维度
- `C`: NMP 通道数量
- `T_K, T_N`: K 和 N 维度的分片因子
- `B_s`: 每通道的 scatter（输入）带宽
- `B_l`: 每通道的 load（输出）带宽

#### 解析解

通过拉格朗日乘数法，可以得到闭式解：

```
T_K = sqrt(C × K × B_l / (N × B_s))
T_N = C / T_K
```

### 2. 数据中心数据流抽象（Data-Centric Dataflow Abstraction）

论文第 5 节提出的数据中心数据流抽象包括三个步骤：

1. **内存访问组划分（MAG Partition）**：将 transformer 层的算子图划分为多个 Memory Access Groups
2. **粗粒度绑定（Coarse-grain Binding）**：将 Memory Partition Groups 分配到内存通道子集
3. **细粒度绑定（Fine-grain Binding）**：确定每个算子层（operator tier）的详细通道分配

## 实现说明

### 文件结构

```
matrixmachine/strategy/
├── h2llm_mapping.py          # H2-LLM 映射策略实现
└── ...
docs/
└── H2LLM_README.md           # 本文档
```

### 主要类

#### `H2LLMTilingStrategy`

实现了论文中的跨通道算子分区算法。

**参数：**
- `element_size`: 矩阵元素大小（字节），默认为 2（FP16）

**注意：** 带宽参数会自动从 chip 对象的 ChipSpec 中提取：
- `scatter_bandwidth`: 从 `chip.spec.die_spec.input_bandwidth` 提取
- `load_bandwidth`: 从 `chip.spec.die_spec.output_bandwidth` 提取

**主要方法：**
- `create_mapping(matrix_shape, chip)`: 创建优化的映射
- `find_optimal_mapping(matrix_shape, chip)`: 创建映射并评估性能

**使用示例：**

```python
from matrixmachine.core.description import MatrixShape
from matrixmachine.workload.hardware.h2llm import create_h2llm_chip
from matrixmachine.strategy.h2llm_mapping import H2LLMTilingStrategy

# 创建硬件配置
chip = create_h2llm_chip(die_count=8)

# 创建策略（带宽参数自动从 chip 中提取）
strategy = H2LLMTilingStrategy()

# 创建映射
matrix = MatrixShape(rows=4096, cols=4096, batch_size=4)
result = strategy.find_optimal_mapping(matrix, chip)

print(f"Latency: {result.latency}")
print(f"Utilization: {result.get_compute_utilization():.2%}")
```

#### `H2LLMDataCentricStrategy`

扩展基础分片策略，实现数据中心数据流抽象（当前版本作为占位符）。

**注意：** 完整的数据中心策略实现需要算子图分析和异构执行映射，这超出了当前实现的范围。

## ✅ 已完成的更新

### 代码改进

**主要变更：**
- ✅ 移除了 `scatter_bandwidth` 和 `load_bandwidth` 作为类属性
- ✅ 带宽参数现在自动从 `Chip` 对象的 `ComputeDieSpec` 中提取
- ✅ 数据格式固定为 FP16（`element_size=2.0`）

**修改的文件：**
1. `matrixmachine/strategy/h2llm_mapping.py`:
   - `H2LLMTilingStrategy` 类简化，只保留 `element_size` 参数
   - `create_mapping()` 方法自动提取带宽：
     ```python
     first_die = list(chip.compute_dies.values())[0]
     scatter_bandwidth = first_die.input_bandwidth  # GB/s
     load_bandwidth = first_die.output_bandwidth    # GB/s
     ```
   - `_calculate_optimal_tiling()` 方法接收带宽作为参数

2. `example_h2llm_mapping.py`:
   - 简化策略创建：`h2llm_strategy = H2LLMTilingStrategy()`
   - 移除了手动传入带宽参数的代码

## 🎯 核心特性

### 自动参数提取
```python
# 旧方式（需要手动传入）
strategy = H2LLMTilingStrategy(
    scatter_bandwidth=12.5,
    load_bandwidth=12.5,
)

# 新方式（自动提取）
strategy = H2LLMTilingStrategy()  # 从 chip 自动提取带宽
```

### 带宽提取逻辑
```python
# 从 chip.spec 直接提取带宽参数
die_spec = chip.spec.die_spec
scatter_bandwidth = die_spec.input_bandwidth   # 输入带宽
load_bandwidth = die_spec.output_bandwidth     # 输出带宽
```

### 假设和约定
- **同构假设**：所有 compute dies 具有相同的配置
- **FP16 数据格式**：元素大小固定为 2 字节
- **带宽单位**：GB/s

## 📝 使用示例

### 基础用法
```python
from matrixmachine.core.description import MatrixShape
from matrixmachine.workload.hardware.h2llm import create_h2llm_chip
from matrixmachine.strategy.h2llm_mapping import H2LLMTilingStrategy

# 创建 H2-LLM 硬件
chip = create_h2llm_chip(die_count=8)

# 创建策略（无需手动配置带宽）
strategy = H2LLMTilingStrategy()

# 创建并评估映射
matrix = MatrixShape(rows=4096, cols=4096, batch_size=4)
result = strategy.find_optimal_mapping(matrix, chip)

print(f"Latency: {result.latency:.2f} cycles")
print(f"Utilization: {result.get_compute_utilization():.2%}")
```

### 自定义配置

```python
from matrixmachine.strategy.h2llm_mapping import H2LLMTilingStrategy
from matrixmachine.workload.hardware.h2llm import create_h2llm_chip
from matrixmachine.core.description import MatrixShape

# 自定义硬件配置
chip = create_h2llm_chip(die_count=4)  # 4 个 compute dies

# 创建策略（带宽参数自动从 chip 提取）
strategy = H2LLMTilingStrategy()

# 测试不同的矩阵大小
matrices = [
    MatrixShape(rows=1024, cols=1024, batch_size=1),
    MatrixShape(rows=2048, cols=2048, batch_size=4),
    MatrixShape(rows=4096, cols=4096, batch_size=16),
]

for matrix in matrices:
    result = strategy.find_optimal_mapping(matrix, chip)
    if result:
        print(f"Matrix: {matrix.rows}×{matrix.cols}×{matrix.batch_size}")
        print(f"  Latency: {result.latency:.2f}")
        print(f"  Utilization: {result.get_compute_utilization():.2%}")
```

### 完整示例
```bash
# 运行示例程序
python example_h2llm_mapping.py
```

## 🔍 技术细节

### H2-LLM 算法（论文第 4.2 节）

**优化目标：**
```
min_{T_K, T_N} s × M × (K/(T_K × B_s) + N/(T_N × B_l))
s.t. T_K × T_N = C
```

**解析解：**
```
T_K = sqrt(C × K × B_l / (N × B_s))
T_N = C / T_K
```

**参数说明：**
- `s`: 元素大小（2 字节，FP16）
- `M, K, N`: 矩阵维度
- `C`: compute die 数量
- `B_s`: scatter 带宽（自动从 `input_bandwidth` 提取）
- `B_l`: load 带宽（自动从 `output_bandwidth` 提取）

## 算法特点

### 1. 解析解优势

- **快速计算**：使用闭式解，无需迭代搜索
- **理论最优**：基于数学推导的最优解
- **参数敏感**：考虑了带宽比例和矩阵维度

### 2. 分片策略

- **M 维度不分片**：避免权重复制（论文第 4.2 节）
- **K/N 维度分片**：根据解析模型确定最优分片
- **批次维度处理**：当有额外容量时分割批次维度

### 3. 负载均衡

- 使用轮询（round-robin）方式分配 tiles
- 确保每个 compute die 获得均衡的工作负载

## 📊 论文结果

根据论文（第 7 节）的实验结果：

- **性能提升**：相比现有 in-die NMP 架构，实现 2.72× 的几何平均加速
- **能效提升**：实现 1.48× 的几何平均能效提升
- **场景覆盖**：在不同批次大小（1/4/16）和不同应用场景下均表现优异

### 测试场景（论文 Table 1）

| 应用场景 | 数据集 | 平均提示长度 | 平均解码长度 |
|---------|-------|------------|------------|
| 代码补全 | HumanEval | 157 | 67 |
| 聊天机器人 | ShareGPT | 783 | 209 |
| 上下文理解 | LongBench | 1886 | 97 |
| 问答 | LooGLE | 1971 | 17 |

## 📊 预期效果（来自论文）

根据 ISCA 2025 论文的实验结果：
- **2.72×** 几何平均加速（相比 in-die NMP）
- **1.48×** 几何平均能效提升
- 适用于不同批次大小（1/4/16）

## 🚀 下一步

建议的扩展方向：
1. **完整的数据中心数据流**：实现 MAG 划分、GCMap 和 OCMap
2. **异构硬件支持**：处理不同 compute dies 具有不同配置的情况
3. **动态调优**：根据运行时反馈调整映射策略
4. **多精度支持**：支持 FP32、INT8 等不同数据格式

## 扩展方向

### 短期扩展

1. **完整的数据中心数据流**：实现 MAG 划分、GCMap 和 OCMap
2. **算子融合支持**：支持多个算子的联合映射
3. **动态调优**：根据运行时反馈调整映射策略

### 长期扩展

1. **自动 DSE 框架**：实现论文第 6 节的设计空间探索框架
2. **多模型支持**：支持不同的 transformer 变体（MHA/GQA/MQA）
3. **性能建模**：集成更精确的性能预测模型

## 📁 文件清单

```
matrixmachine/strategy/
├── h2llm_mapping.py          # 核心实现（已更新）
└── __init__.py               # 模块导出（已更新）

docs/
└── H2LLM_README.md           # 本文档

example_h2llm_mapping.py      # 示例程序（已更新）
```

## ✨ 关键优势

1. **更简洁的 API**：用户无需关心带宽配置细节
2. **自动适配**：根据硬件配置自动调整
3. **类型安全**：FP16 格式固定，避免配置错误
4. **易于维护**：减少了配置参数，降低出错概率

## 参考文献

```bibtex
@inproceedings{h2llm2025,
  title={H2-LLM: Hardware-Dataflow Co-Exploration for Heterogeneous Hybrid-Bonding-based Low-Batch LLM Inference},
  author={Li, Cong and Yin, Yihan and Wu, Xintong and Zhu, Jingchen and Gao, Zhutianya and Niu, Dimin and Wu, Qiang and Si, Xin and Xie, Yuan and Zhang, Chen and Sun, Guangyu},
  booktitle={Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA)},
  year={2025}
}
```

## 相关链接

- 论文 PDF: `papers/H2-LLM.pdf`
- 开源代码: https://github.com/leesong/H2-LLM-ISCA-2025
- 示例代码: `example_h2llm_mapping.py`

## 贡献者

实现基于 MatrixMachine 框架和 H2-LLM 论文。

---

**实现完成日期：** 2025-01-XX
**基于论文：** H2-LLM (ISCA 2025)
**框架版本：** MatrixMachine v0.1