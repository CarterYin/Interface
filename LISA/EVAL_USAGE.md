# LISA模型评估使用指南

本指南展示如何使用 `eval.py` 对LISA模型进行RefCOCO和ReasonSeg数据集评估。

## 快速开始

### 1. 准备工作

```bash
# 确保已安装必要依赖
pip install torch torchvision pillow numpy opencv-python transformers

# 将eval.py复制到LISA文件夹（或添加到Python路径）
cp ../eval.py .
```

### 2. 最简单的使用方式

创建一个简单的评估脚本 `run_eval.py`:

```python
import sys
import os

# 导入eval.py中的函数
from eval import ReasonSegDataset, predict_model_by_single_sample, compute_metrics

# 导入LISA模型（这里需要实现load_model和forward_single_sample）
# 见下面的完整示例

# 1. 加载模型
model = load_model()

# 2. 加载数据集
dataset = ReasonSegDataset("../ReasonSeg/val")

# 3. 运行预测
results = predict_model_by_single_sample(model, dataset)

# 4. 计算指标
metrics = compute_metrics(dataset, results)
print(metrics)
```

### 3. 完整示例

查看 `eval_lisa_example.py` 获取完整的实现示例，包括：
- 如何加载LISA模型
- 如何实现 `forward_single_sample` 函数
- 如何处理分割任务和定位任务

运行完整示例：

```bash
cd LISA
python eval_lisa_example.py
```

## 关键函数实现

### load_model 函数

```python
def load_model(version="xinlai/LISA-13B-llama2-v1", **kwargs):
    """加载LISA模型"""
    from model.LISA import LISAForCausalLM
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(version)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    model = LISAForCausalLM.from_pretrained(
        version,
        low_cpu_mem_usage=True,
        vision_tower="openai/clip-vit-large-patch14",
        seg_token_idx=seg_token_idx,
    )
    
    model.get_model().initialize_vision_modules(model.get_model().config)
    model = model.cuda().eval()
    
    return model
```

### forward_single_sample 函数

```python
def forward_single_sample(model, example):
    """
    LISA单样本推理
    
    Args:
        model: LISA模型
        example: dict包含 'image' (PIL Image) 和 'text' (str)
    
    Returns:
        dict包含 'mask' (numpy array) 用于ReasonSeg
        或 'answer' (str) 用于RefCOCO
    """
    image = example['image']  # PIL Image
    text = example['text']     # str
    
    # 1. 预处理图像和文本
    # ... (见完整示例)
    
    # 2. 模型推理
    with torch.no_grad():
        output_ids, pred_masks = model.evaluate(...)
    
    # 3. 后处理
    pred_mask = pred_masks[0].cpu().numpy() if pred_masks else None
    
    return {'mask': pred_mask}  # 或 {'answer': text_output}
```

## 数据集准备

### ReasonSeg数据集

```
E:\eval\ReasonSeg\val\
├── image001.jpg
├── image001.json
├── image002.jpg
├── image002.json
└── ...
```

### RefCOCO数据集

创建JSONL格式文件 `refcoco_val.jsonl`:

```json
{"image": "path/to/img.jpg", "sent": "the red car", "bbox": [100, 100, 200, 200], "width": 640, "height": 480}
{"image": "path/to/img2.jpg", "sent": "person on left", "bbox": [50, 50, 150, 300], "width": 640, "height": 480}
```

## 评估指标

### ReasonSeg指标
- **P@0.5 ~ P@0.9**: 不同IoU阈值下的精度
- **mean_iou**: 平均IoU

### RefCOCO指标
- **Precision@0.5**: IoU≥0.5的比例
- **mean_iou**: 平均IoU

## 常见问题

### Q1: 显存不足怎么办？

使用半精度或8bit量化：

```python
model = load_model(precision="fp16")  # 或 "bf16"
# 或使用8bit量化
model = load_model(load_in_8bit=True)
```

### Q2: 如何只测试部分数据？

```python
# 创建子集
class SubDataset:
    def __init__(self, dataset, num_samples):
        self.dataset = dataset
        self.num = min(num_samples, len(dataset))
    
    def __len__(self):
        return self.num
    
    def __getitem__(self, idx):
        return self.dataset[idx]

small_dataset = SubDataset(dataset, num_samples=100)
results = predict_model_by_single_sample(model, small_dataset)
```

### Q3: 如何保存预测结果？

```python
import json

# 保存结果
with open('predictions.json', 'w') as f:
    json.dump(results, f)

# 后续可以直接加载计算指标
with open('predictions.json', 'r') as f:
    results = json.load(f)
metrics = compute_metrics(dataset, results)
```

## 模型权重

### SAM 权重（本地）

**位置**: `LISA/sam/sam_vit_h_4b8939.pth`

`eval_lisa_example.py` 会自动使用该本地 SAM 权重文件。确保该文件存在：

```bash
# 检查 SAM 权重是否存在
ls LISA/sam/sam_vit_h_4b8939.pth

# 如果不存在，从以下位置下载：
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

代码中自动设置：
```python
# 脚本会自动使用这个路径
SAM_CHECKPOINT_PATH = "LISA/sam/sam_vit_h_4b8939.pth"
```

### LISA 主模型

完整 LISA 模型需要从 Hugging Face 下载（会自动保存到 `model_download/` 目录）：

```bash
# 方式1: 使用git lfs下载到本地
git lfs clone https://huggingface.co/xinlai/LISA-13B-llama2-v1

# 然后在代码中指定本地路径
model = load_model(version="/path/to/LISA-13B-llama2-v1")

# 方式2: 在代码中自动下载（需要联网，会保存到 model_download/）
model = load_model(version="xinlai/LISA-13B-llama2-v1")
```

### 自定义 SAM 权重路径

如果 SAM 权重在其他位置，可以在调用 `load_model` 时指定：

```python
model = load_model(
    version="xinlai/LISA-13B-llama2-v1",
    sam_checkpoint="/path/to/your/sam_vit_h_4b8939.pth"
)
```

## 更多信息

- eval.py 完整文档: `../README.md`
- LISA项目: `LISA/README.md`
- 完整示例代码: `eval_lisa_example.py`

