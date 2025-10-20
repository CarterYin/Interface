[//]: # (<br />)
<p align="center">
  <h1 align="center">几个接口的实现</h1>
  <p align="center">
    <img src="https://img.shields.io/badge/Interface-blue?style=flat&logo=github" alt="Interface">
    <img src="https://img.shields.io/badge/Python%20%7C%20Interface-green" alt="Python">
  </p>
</p>

## 要求

这边需要麻烦你封装几个数据集的 eval 代码，方便后续扩展到不同模型和数据集。这里的 RefCOCO 你之前复现过的代码应该是有这部分的，ReasonSeg 我也给了对应的链接。

我在文件里定义了几个接口，主要就是实现这些接口。模型的话可以先自己定义一个容易验证的，或者用你之前复现过的。也可以多找找看其他人有没有类似的实现，然后拿过来改一改。

引入的依赖尽可能少，这样对不同的模型，都可以直接把文件放到他们的项目里，然后 import 他们的模型直接测。

## 实际需要完成的部分

这段代码提供了一个机器学习评估流程的**框架 (framework)**，但其中许多关键的功能逻辑是**缺失的**或**仅为占位符**。

为了使这段代码能够实际运行并完成一个有意义的模型评估，你需要完成以下几个部分：

***

### 1. 核心数据结构和方法 (Core Data Structures & Methods)

这是整个 PyTorch 数据管道的基础。

| 接口 | 需完成的功能 | 说明 |
| :--- | :--- | :--- |
| `RefCOCODataset`, `ReasonSegDataset` | 实现 **`__len__`** 和 **`__getitem__`** 方法 | 继承自 `torch.utils.data.Dataset` 的类**必须**实现这两个方法。`__len__` 返回数据集大小，`__getitem__(idx)` 返回索引为 `idx` 的单个数据样本及其标签（通常为 `Dict`）。 |
| `load_model` | 实现模型加载逻辑 | **实例化** PyTorch 模型（`nn.Module`），并可选地加载预训练权重。 |

***

### 2. 核心模型推理逻辑 (Core Model Inference Logic)

这是模型产生预测结果的实际计算部分。

| 接口 | 需完成的功能 | 说明 |
| :--- | :--- | :--- |
| `forward_single_sample` | 实现单样本前向传播 | 将单个样本（`Dict`）转换为模型所需的 **Tensor** 格式（通常需增加 Batch 维度，例如 `input.unsqueeze(0)`），调用模型 (`model(input)`), 并将输出（Tensor）转换回 **`Dict`** 结果格式。 |
| `forward_batch_samples` | 实现批量前向传播 | 这是一个更复杂的步骤。需将 `List[Dict]` 样本通过**数据整理/批次化 (Collation)** 转换为单个或多个 **Batch Tensor**，调用模型 (`model(batch_input)`), 最后将输出的 Tensor 结果解析回 `List[Dict]` 格式。|

***

### 3. 结果评估逻辑 (Result Evaluation Logic)

这是将预测结果转化为可理解的性能指标的部分。

| 接口 | 需完成的功能 | 说明 |
| :--- | :--- | :--- |
| `compute_metrics` | 实现指标计算逻辑 | **对比** `dataset` 中的真实标签 (Ground Truth) 和 `results` (模型的预测值)，计算如 IoU、准确率、F1 分数等与 RefCOCO/ReasonSeg 任务相关的**性能指标**。|

***

### 总结

总而言之，这段代码为你定义了**流程 (What to do)** 和**接口 (How to connect)**，但你需要填充所有带 `# Implement ... logic here` 注释的函数和数据集类的核心方法，以完成**实际的计算 (How it works)**。
* **必须完成：** `__len__`, `__getitem__`, `load_model`, `forward_single_sample`, `forward_batch_samples`, `compute_metrics` 的实现。