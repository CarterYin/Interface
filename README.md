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


## 许可和贡献

欢迎提出 Issue 或 Pull Request！

如有问题或建议，欢迎联系。
