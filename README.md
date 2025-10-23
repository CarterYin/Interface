[//]: # (<br />)
<p align="center">
  <h1 align="center">RefCOCO/ReasonSeg 数据集评估框架</h1>
  <p align="center">
    <img src="https://img.shields.io/badge/Interface-blue?style=flat&logo=github" alt="Interface">
    <img src="https://img.shields.io/badge/Python%20%7C%20Interface-green" alt="Python">
  </p>
</p>

## 项目简介

这是一个封装好的 RefCOCO/ReasonSeg 数据集评估框架，方便后续扩展到不同模型和数据集。

**设计理念：**  
引入的依赖尽可能少，这样对不同的模型，都可以直接把文件放到他们的项目里，然后 import 他们的模型直接测。

---

## 依赖项

最小依赖（适用于大多数深度学习项目）：

```bash
pip install torch pillow numpy opencv-python
```

## 快速开始

### 1. RefCOCO 评估（边界框定位任务）

```python
from eval import RefCOCODataset, load_model, predict_model_by_single_sample, compute_metrics

# 1. 加载数据集
dataset = RefCOCODataset(
    jsonl_path="data/refcoco/refcoco_val.jsonl",
    prompt_template="Please provide the bounding box coordinate of the region this sentence describes: {}"
)

# 2. 加载你的模型
def load_model(**kwargs):
    from your_model import YourModel  # 替换为你的模型
    model = YourModel.from_pretrained("path/to/checkpoint")
    model.eval()
    return model

model = load_model()

# 3. 实现单样本推理函数
def forward_single_sample(model, example):
    image = example['image']  # PIL Image
    text = example['text']    # str
    
    # 调用你的模型进行推理
    # 模型应该输出包含边界框坐标的文本，格式如 "[x1,y1,x2,y2]"
    output = model.generate(image, text)
    
    return {'answer': output}

# 4. 运行预测
results = predict_model_by_single_sample(model, dataset)

# 5. 计算评估指标
metrics = compute_metrics(dataset, results)
print(metrics)
# 输出示例: {'Precision@0.5': 0.85, 'mean_iou': 0.72, 'total_samples': 1000}
```

### 2. ReasonSeg 评估（语义分割任务）

```python
from eval import ReasonSegDataset, load_model, predict_model_by_single_sample, compute_metrics

# 1. 加载数据集
dataset = ReasonSegDataset(image_folder="ReasonSeg/val")

# 2. 加载你的模型
model = load_model()

# 3. 实现单样本推理函数
def forward_single_sample(model, example):
    image = example['image']  # PIL Image
    text = example['text']    # str
    
    # 调用你的模型进行推理
    # 模型应该输出分割掩码，numpy array (H, W)
    mask = model.segment(image, text)
    
    return {'mask': mask}

# 4. 运行预测
results = predict_model_by_single_sample(model, dataset)

# 5. 计算评估指标
metrics = compute_metrics(dataset, results)
print(metrics)
# 输出示例: {'P@0.5': 0.80, 'P@0.7': 0.65, 'mean_iou': 0.68, ...}
```

### 3. 批量推理（提高效率）

```python
from eval import predict_model_by_batch_samples

# 实现批量推理函数
def forward_batch_samples(model, examples):
    images = [ex['image'] for ex in examples]
    texts = [ex['text'] for ex in examples]
    
    # 调用你的模型进行批量推理
    outputs = model.batch_generate(images, texts)
    
    return [{'answer': out} for out in outputs]

# 使用批量预测
results = predict_model_by_batch_samples(model, dataset, batch_size=16)
metrics = compute_metrics(dataset, results)
print(metrics)
```

## 数据格式说明

### RefCOCO 数据格式

JSONL 文件，每行一个 JSON 对象：

```json
{
    "image": "path/to/image.jpg",
    "sent": "the person on the left",
    "bbox": [120, 50, 300, 400],
    "width": 640,
    "height": 480
}
```

- `image`: 图像文件路径
- `sent`: 指称表达式文本
- `bbox`: 边界框坐标 `[x1, y1, x2, y2]`，单位为像素
- `width`, `height`: 图像宽度和高度

### ReasonSeg 数据格式

文件夹结构：

```
ReasonSeg/val/
├── image1.jpg
├── image1.json
├── image2.jpg
├── image2.json
└── ...
```

JSON 标注文件格式：

```json
{
    "shapes": [
        {
            "label": "object",
            "points": [[x1, y1], [x2, y2], ..., [xn, yn]]
        }
    ],
    "text": "description of the object"
}
```

## 评估指标说明

### RefCOCO 指标

- **Precision@0.5**: IoU ≥ 0.5 的样本比例
- **mean_iou**: 平均 IoU
- **total_samples**: 总样本数

### ReasonSeg 指标

- **P@0.5, P@0.6, P@0.7, P@0.8, P@0.9**: 不同 IoU 阈值下的精度
- **mean_iou**: 平均 IoU
- **total_samples**: 总样本数

## 核心函数说明

### 必须实现的函数

1. **`load_model(**kwargs)`**: 加载你的模型
2. **`forward_single_sample(model, example)`**: 单样本推理
   - 输入：`example` 包含 `image` (PIL Image) 和 `text` (str)
   - 输出：对于 RefCOCO 返回 `{'answer': text_with_bbox}`，对于 ReasonSeg 返回 `{'mask': numpy_array}`
3. **`forward_batch_samples(model, examples)`**: 批量推理（可选，默认会循环调用单样本推理）

### 预定义的函数

- **`predict_model_by_single_sample(model, dataset)`**: 逐样本预测
- **`predict_model_by_batch_samples(model, dataset, batch_size)`**: 批量预测
- **`compute_metrics(dataset, results)`**: 计算评估指标

## 使用示例

完整的评估流程示例：

```python
import eval

# 修改 eval.py 中的 load_model 函数
def load_model(**kwargs):
    from llava.model.builder import load_pretrained_model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path="path/to/llava-v1.5-7b",
        model_base=None,
        model_name="llava-v1.5-7b"
    )
    return model

# 修改 eval.py 中的 forward_single_sample 函数
def forward_single_sample(model, example):
    # 你的推理代码
    ...
    return {'answer': output}

# 运行评估
if __name__ == "__main__":
    eval.main()
```

## 完整示例

将 `eval.py` 复制到你的项目中，修改 `load_model` 和 `forward_single_sample` 函数，然后运行：

```bash
python eval.py
```

或者在你的代码中导入使用：

```python
from eval import RefCOCODataset, compute_metrics
# ... 你的代码
```

---

