[//]: # (<br />)
<p align="center">
  <h1 align="center">RefCOCO 数据集评估框架</h1>
  <p align="center">
    <img src="https://img.shields.io/badge/Interface-blue?style=flat&logo=github" alt="Interface">
    <img src="https://img.shields.io/badge/Python%20%7C%20Interface-green" alt="Python">
  </p>
</p>

## 项目简介

这是一个封装好的 RefCOCO/ReasonSeg 数据集评估框架，方便后续扩展到不同模型和数据集。

**主要特性：**
- ✅ 自动加载 RefCOCO 数据集（使用 HuggingFace datasets）
- ✅ 支持单样本和批量预测
- ✅ 完整的评估指标计算（IoU, Precision, Recall, Accuracy）
- ✅ 依赖最小化，易于集成到不同项目

**设计理念：**  
引入的依赖尽可能少，这样对不同的模型，都可以直接把文件放到他们的项目里，然后 import 他们的模型直接测。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖项：
```
torch>=1.10.0
numpy>=1.19.0
datasets>=2.0.0
Pillow>=8.0.0
```

### 2. 运行评估

```bash
# 使用完整的 val 集
python eval.py

# 只使用前 10 个样本快速测试
python eval.py --max-samples 10

# 使用不同的数据集划分
python eval.py --split test
python eval.py --split testA
python eval.py --split testB
```

**输出示例：**
```
=== RefCOCO 数据集评估 ===
正在加载 RefCOCO 数据集: refcoco, split: val
数据集加载完成，共 8811 个样本
数据集大小: 8811
模型类型: SimpleSegmentationModel

=== 单样本预测 ===
预测结果数量: 8811

=== 批量预测 ===
预测结果数量: 8811

=== 单样本指标 ===
mean_iou: 0.4532
overall_iou: 0.4621
precision: 0.5123
recall: 0.4891
accuracy: 0.4123

=== 批量指标 ===
mean_iou: 0.4532
overall_iou: 0.4621
precision: 0.5123
recall: 0.4891
accuracy: 0.4123
```

---
<!-- 
## 使用指南

### 方式 1: 直接使用 HuggingFace datasets

```python
from datasets import load_dataset

# 加载 RefCOCO 数据集
ds = load_dataset("lmms-lab/RefCOCO", split="val")

print(f"数据集大小: {len(ds)}")
print(f"第一个样本: {ds[0].keys()}")
```

### 方式 2: 使用封装的 RefCOCODataset 类（推荐）

```python
from eval import RefCOCODataset

# 创建数据集
dataset = RefCOCODataset(
    split="val",         # 或 'test', 'testA', 'testB'
    image_size=224       # 图像调整大小
)

# 获取样本
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")  # [C, H, W]
print(f"Text: {sample['text']}")
print(f"Mask shape: {sample['mask'].shape}")    # [H, W]
print(f"Image ID: {sample['image_id']}")
```

### 方式 3: 集成到你的模型评估中

```python
from eval import (
    RefCOCODataset,
    load_model,
    predict_model_by_batch_samples,
    compute_metrics
)

# 1. 加载数据集
dataset = RefCOCODataset(split="val")

# 2. 加载你的模型（替换为实际模型）
def load_model(**kwargs):
    from your_project.models import YourSegmentationModel
    
    model = YourSegmentationModel()
    model.load_state_dict(torch.load("path/to/checkpoint.pth"))
    model.eval()
    return model

model = load_model()

# 3. 进行预测
results = predict_model_by_batch_samples(model, dataset, batch_size=16)

# 4. 计算评估指标
metrics = compute_metrics(dataset, results)

print("评估结果:")
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")
``` 

---


## 数据格式说明

### 输入数据格式（从数据集获取）

每个样本是一个字典，包含以下键：

```python
{
    "image": torch.Tensor,  # 形状 [C, H, W]，值范围 [0, 1]
    "text": str,            # 文本描述，例如 "the red car on the left"
    "mask": torch.Tensor,   # 形状 [H, W]，二值掩码（0 或 1）
    "image_id": str,        # 图像唯一标识符
    "bbox": list,           # 可选，边界框 [x, y, w, h]
}
```

### 模型输出格式

模型应该输出分割掩码：

```python
def your_model_forward(model, images, texts):
    """
    Args:
        images: torch.Tensor, shape [B, C, H, W]
        texts: List[str], 长度为 B
    
    Returns:
        masks: torch.Tensor, shape [B, H, W]
        值范围 [0, 1]，可以是 logits 或概率
    """
    masks = model(images, texts)
    return masks
```

---
-->


## 评估指标说明

代码计算以下指标：

1. **mean_iou** (平均 IoU)
   - 每个样本的 IoU 的平均值
   - 公式: IoU = Intersection / Union
   - 范围: [0, 1]，越高越好

2. **overall_iou** (整体 IoU)
   - 所有样本的总交集除以总并集
   - 对大目标更敏感

3. **precision** (精确率)
   - 预测为正例中真正为正例的比例
   - 公式: Precision = TP / (TP + FP)

4. **recall** (召回率)
   - 真实正例中被正确预测的比例
   - 公式: Recall = TP / (TP + FN)

5. **accuracy** (准确率)
   - IoU > 0.5 的样本比例
   - 这是 RefCOCO 常用的评估指标

---

## 自定义模型接口

### 选项 1: 修改 load_model 函数

在 `eval.py` 中找到 `load_model` 函数：

```python
def load_model(**kwargs):
    from your_project.models import YourModel
    
    model = YourModel()
    # 加载权重
    if "model_path" in kwargs:
        model.load_state_dict(torch.load(kwargs["model_path"]))
    
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model
```

### 选项 2: 修改 forward 函数

如果你的模型接口不同，可以修改 `forward_batch_samples` 函数：

```python
def forward_batch_samples(model, examples):
    device = next(model.parameters()).device
    
    # 根据你的模型需求准备输入
    images = torch.stack([ex["image"] for ex in examples]).to(device)
    texts = [ex["text"] for ex in examples]
    
    # 调用你的模型
    outputs = model.inference(images=images, prompts=texts)
    pred_masks = outputs["segmentation"]  # 根据实际返回格式调整
    
    # 构建结果
    predictions = []
    for i in range(len(examples)):
        predictions.append({
            "mask": pred_masks[i].cpu(),
            "image_id": examples[i]["image_id"],
        })
    
    return predictions
```

---

## RefCOCO 数据集说明

### 数据集信息

- **来源**: `lmms-lab/RefCOCO` (HuggingFace)
- **任务**: 指代表达式分割（Referring Expression Segmentation）
- **可用划分**:
  - `val`: 验证集（8,811 个样本）
  - `test`: 测试集（5,000 个样本）
  - `testA`: 测试集 A（1,975 个样本，人物相关）
  - `testB`: 测试集 B（1,810 个样本，物体相关）

### 加载方式

```python
from datasets import load_dataset

# 方式 1: 直接使用 datasets 库
ds = load_dataset("lmms-lab/RefCOCO", split="val")

# 方式 2: 使用我们封装的 RefCOCODataset 类（推荐）
from eval import RefCOCODataset
dataset = RefCOCODataset(split="val")
```

**注意：** 首次运行时会自动从 HuggingFace 下载数据集（约几 GB），之后会使用缓存。

---

<!-- ## 完整示例

```python
import torch
from eval import RefCOCODataset, compute_metrics

# 1. 准备数据集
dataset = RefCOCODataset(split="val")

# 2. 加载你的模型
from your_project import YourModel
model = YourModel()
model.load_state_dict(torch.load("checkpoint.pth"))
model.eval()
model = model.cuda()

# 3. 进行推理
predictions = {"predictions": []}

for i, sample in enumerate(dataset):
    image = sample["image"].unsqueeze(0).cuda()
    text = sample["text"]
    
    with torch.no_grad():
        pred_mask = model(image, [text])  # [1, H, W]
    
    predictions["predictions"].append({
        "mask": pred_mask.squeeze(0).cpu(),
        "image_id": sample["image_id"],
    })
    
    if (i + 1) % 100 == 0:
        print(f"已处理 {i + 1}/{len(dataset)} 个样本")

# 4. 计算指标
metrics = compute_metrics(dataset, predictions)

print("\n最终评估结果:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
```

--- -->

<!-- ## 常见问题

### Q1: 首次运行很慢？

**A:** 首次运行时会从 HuggingFace 下载 RefCOCO 数据集，可能需要几分钟。之后会使用缓存，速度会快很多。

### Q2: 如何只测试部分数据？

**A:** 使用 `--max-samples` 参数：
```bash
python eval.py --max-samples 10
```

### Q3: 数据集保存在哪里？

**A:** HuggingFace datasets 会将数据缓存在 `~/.cache/huggingface/datasets/` 目录下。

### Q4: 如何使用自己的模型？

**A:** 修改 `eval.py` 中的 `load_model` 函数，导入并加载你的模型即可。详见"自定义模型接口"部分。

### Q5: 支持哪些数据集？

**A:** 目前实现了：
- RefCOCO（通过 HuggingFace datasets 加载）
- ReasonSeg（预留接口，需要自行实现加载逻辑）

### Q6: 如何扩展到新数据集？

**A:** 参照 `RefCOCODataset` 和 `ReasonSegDataset` 的实现，创建新的数据集类。主要需要实现：
- `__init__`: 初始化和数据加载
- `__len__`: 返回数据集大小
- `__getitem__`: 返回单个样本

--- -->

## 文件结构

```
eval/
├── eval.py              # 主代码文件
├── requirements.txt     # 依赖列表
└── README.md           # 本文档
```

---

<!-- ## 主要接口说明

### 数据集接口
- `RefCOCODataset`: RefCOCO 数据集类
  - `__len__()`: 返回数据集大小
  - `__getitem__(idx)`: 返回第 idx 个样本
- `ReasonSegDataset`: ReasonSeg 数据集类（预留）

### 模型推理接口
- `load_model(**kwargs)`: 加载或初始化模型
- `forward_single_sample(model, example)`: 单样本前向传播
- `forward_batch_samples(model, examples)`: 批量前向传播

### 预测接口
- `predict_model_by_single_sample(model, dataset)`: 单样本预测
- `predict_model_by_batch_samples(model, dataset, batch_size)`: 批量预测

### 评估接口
- `compute_metrics(dataset, results)`: 计算评估指标

--- -->

<!-- ## 更新日志

### v1.0 - 当前版本

**主要特性：**
- ✅ 集成 HuggingFace datasets 库自动加载 RefCOCO
- ✅ 支持命令行参数灵活控制
- ✅ 完善的数据预处理（图像缩放、归一化、掩码处理）
- ✅ 支持单样本和批量推理
- ✅ 完整的评估指标计算

**依赖：**
- `torch>=1.10.0`
- `numpy>=1.19.0`
- `datasets>=2.0.0`
- `Pillow>=8.0.0`

**命令行参数：**
```bash
python eval.py [--split SPLIT] [--max-samples N]

选项:
  --split SPLIT       数据集划分，可选: val, test, testA, testB（默认: val）
  --max-samples N     最大样本数量，用于快速测试
```

--- -->

## 许可和贡献

欢迎提出 Issue 或 Pull Request！

如有问题或建议，欢迎联系。
