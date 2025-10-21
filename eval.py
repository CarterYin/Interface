import os
from typing import Dict, List, Union, Optional
import json

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image


class RefCOCODataset(Dataset):
    """RefCOCO/RefCOCO+/RefCOCOg 数据集类
    
    用于指代表达式分割任务（Referring Expression Segmentation）
    使用 HuggingFace datasets 库自动加载数据
    
    Args:
        split: 数据集划分，如 'val', 'test', 'testA', 'testB'
        dataset_name: 数据集名称，'refcoco', 'refcoco+', 'refcocog'
        image_size: 图像大小，用于调整图像和掩码尺寸（默认 224）
    """
    
    def __init__(
        self,
        split: str = "val",
        dataset_name: str = "refcoco",
        image_size: int = 224,
    ):
        super().__init__()
        self.split = split
        self.dataset_name = dataset_name
        self.image_size = image_size
        
        # 使用 HuggingFace datasets 加载数据
        print(f"正在加载 RefCOCO 数据集: {dataset_name}, split: {split}")
        self.dataset = load_dataset("lmms-lab/RefCOCO", split=split)
        print(f"数据集加载完成，共 {len(self.dataset)} 个样本")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个数据样本
        
        Returns:
            Dict: 包含以下键的字典
                - image: 图像张量 [C, H, W]
                - text: 文本描述
                - mask: 真实分割掩码 [H, W]
                - image_id: 图像ID
                - bbox: 边界框（可选）
        """
        # 从 HuggingFace 数据集获取样本
        sample = self.dataset[idx]
        
        # 处理图像
        image = sample.get('image', sample.get('img'))
        if isinstance(image, Image.Image):
            # 确保是 RGB 模式（3 通道）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # PIL Image 转换为 tensor
            image = image.resize((self.image_size, self.image_size))
            image = torch.from_numpy(np.array(image)).float()
            # [H, W, C] -> [C, H, W]
            if len(image.shape) == 3:
                image = image.permute(2, 0, 1)
            elif len(image.shape) == 2:
                # 灰度图，扩展到 3 通道
                image = image.unsqueeze(0).repeat(3, 1, 1)
            # 归一化到 [0, 1]
            if image.max() > 1:
                image = image / 255.0
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = image.permute(2, 0, 1)
            elif len(image.shape) == 2:
                # 灰度图，扩展到 3 通道
                image = image.unsqueeze(0).repeat(3, 1, 1)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # 单通道图像，扩展到 3 通道
                image = image.permute(2, 0, 1).repeat(3, 1, 1)
            if image.max() > 1:
                image = image / 255.0
        
        # 最终确保是 3 通道
        if len(image.shape) == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        # 处理文本
        text = sample.get('text', sample.get('caption', sample.get('sentence', '')))
        if isinstance(text, list) and len(text) > 0:
            text = text[0]  # 如果是列表，取第一个
        
        # 处理掩码
        mask = sample.get('mask', sample.get('segmentation'))
        if mask is not None:
            if isinstance(mask, Image.Image):
                mask = mask.resize((self.image_size, self.image_size))
                mask = torch.from_numpy(np.array(mask)).float()
            elif isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float()
            elif isinstance(mask, list):
                # 如果是列表，转换为 numpy 数组
                mask = np.array(mask)
                mask = torch.from_numpy(mask).float()
            elif isinstance(mask, torch.Tensor):
                mask = mask.float()
            else:
                # 其他类型，尝试转换
                mask = torch.tensor(mask).float()
            
            # 确保掩码是 2D 的（在二值化之前）
            if len(mask.shape) == 3:
                mask = mask[0]  # 取第一个通道
            elif len(mask.shape) == 1:
                # 如果是 1D，尝试 reshape
                size = int(np.sqrt(len(mask)))
                if size * size == len(mask):
                    mask = mask.reshape(size, size)
                else:
                    # 无法 reshape，创建零掩码
                    mask = torch.zeros(self.image_size, self.image_size).float()
            
            # 调整大小到目标尺寸
            if mask.shape[0] != self.image_size or mask.shape[1] != self.image_size:
                mask = torch.from_numpy(
                    np.array(Image.fromarray(mask.numpy()).resize((self.image_size, self.image_size)))
                ).float()
            
            # 二值化
            mask = (mask > 0).float()
        else:
            # 如果没有掩码，创建一个零掩码
            mask = torch.zeros(self.image_size, self.image_size).float()
        
        # 处理 image_id
        image_id = sample.get('image_id', sample.get('img_id', f"image_{idx}"))
        if not isinstance(image_id, str):
            image_id = str(image_id)
        
        # 处理边界框
        bbox = sample.get('bbox', None)
        
        return {
            "image_id": image_id,
            "image": image,
            "text": text,
            "mask": mask,
            "bbox": bbox,
        }


class ReasonSegDataset(Dataset):
    """ReasonSeg 数据集类
    
    ReasonSeg 是一个复杂推理分割数据集，需要更复杂的推理能力
    TODO: 需要实现具体的数据加载逻辑
    
    Args:
        data_root: 数据根目录
        split: 数据集划分，如 'val', 'test'
        image_size: 图像大小
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "val",
        image_size: int = 224,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        
        # TODO: 实现 ReasonSeg 数据加载
        # 示例：
        # self.dataset = load_dataset("path/to/reasonseg", split=split)
        raise NotImplementedError("ReasonSeg 数据集加载尚未实现，请参考 RefCOCODataset 实现")
    
    def __len__(self) -> int:
        raise NotImplementedError("ReasonSeg 数据集尚未实现")
    
    def __getitem__(self, idx: int) -> Dict:
        raise NotImplementedError("ReasonSeg 数据集尚未实现")


class SimpleSegmentationModel(nn.Module):
    """简单的分割模型用于测试
    
    这是一个极简的模型，实际使用时应该替换为真实的模型
    如 LISA, PixelLM, LAVT 等
    """
    
    def __init__(self, image_size: int = 224):
        super().__init__()
        self.image_size = image_size
        
        # 简单的卷积网络
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),  # 输出单通道掩码
        )
    
    def forward(self, images: torch.Tensor, texts: Optional[List[str]] = None) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] 图像张量
            texts: 文本描述列表（此简单模型未使用）
        
        Returns:
            masks: [B, H, W] 预测的分割掩码
        """
        # [B, C, H, W] -> [B, 1, H, W]
        logits = self.encoder(images)
        # [B, 1, H, W] -> [B, H, W]
        masks = torch.sigmoid(logits).squeeze(1)
        return masks


def main(split: str = "val", max_samples: int = None):
    """主函数示例
    
    Args:
        split: 数据集划分，可选: 'val', 'test', 'testA', 'testB'
        max_samples: 最大样本数量，用于快速测试（None 表示使用全部数据）
    """
    # 创建数据集
    print("=== RefCOCO 数据集评估 ===")
    dataset = RefCOCODataset(split=split)
    print(f"数据集大小: {len(dataset)}")
    
    # 如果设置了 max_samples，则只使用部分数据
    if max_samples is not None and max_samples < len(dataset):
        print(f"为了快速测试，只使用前 {max_samples} 个样本")
        # 创建一个子集
        from torch.utils.data import Subset
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)
        print(f"使用数据集大小: {len(dataset)}")
    
    # 加载模型
    model = load_model()
    print(f"模型类型: {type(model).__name__}")
    
    # 进行预测
    print("\n=== 单样本预测 ===")
    single_sample_results = predict_model_by_single_sample(model, dataset)
    print(f"预测结果数量: {len(single_sample_results['predictions'])}")
    
    print("\n=== 批量预测 ===")
    batch_sample_results = predict_model_by_batch_samples(model, dataset, batch_size=16)
    print(f"预测结果数量: {len(batch_sample_results['predictions'])}")
    
    # 计算指标
    print("\n=== 单样本指标 ===")
    metrics_by_single_samples = compute_metrics(dataset, single_sample_results)
    for key, value in metrics_by_single_samples.items():
        print(f"{key}: {value:.4f}")
    
    print("\n=== 批量指标 ===")
    metrics_by_batch_samples = compute_metrics(dataset, batch_sample_results)
    for key, value in metrics_by_batch_samples.items():
        print(f"{key}: {value:.4f}")


def load_model(**kwargs) -> nn.Module:
    """加载或初始化模型
    
    Args:
        model_path: 模型权重路径（可选）
        model_type: 模型类型（可选）
        device: 运行设备（可选）
    
    Returns:
        model: PyTorch 模型
    """
    model_path = kwargs.get("model_path", None)
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = SimpleSegmentationModel()
    
    # 加载预训练权重（如果提供）
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"从 {model_path} 加载模型权重")
    
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    return model


def compute_metrics(
    dataset: Union[RefCOCODataset, ReasonSegDataset, "torch.utils.data.Subset"],
    results: Dict[str, List[Dict]]
) -> Dict[str, float]:
    """计算评估指标
    
    Args:
        dataset: 数据集对象，包含真实标签
        results: 预测结果字典，格式为 {"predictions": [{"mask": ...}, ...]}
    
    Returns:
        metrics: 包含各项指标的字典
            - mean_iou: 平均 IoU
            - overall_iou: 整体 IoU
            - precision: 精确率
            - recall: 召回率
            - accuracy: 准确率（IoU > 0.5）
    """
    predictions = results["predictions"]
    
    if len(predictions) != len(dataset):
        raise ValueError(f"预测数量 ({len(predictions)}) 与数据集大小 ({len(dataset)}) 不匹配")
    
    ious = []
    precisions = []
    recalls = []
    
    total_intersection = 0
    total_union = 0
    
    for idx, pred_dict in enumerate(predictions):
        # 获取真实标签
        gt_sample = dataset[idx]
        gt_mask = gt_sample["mask"]
        
        # 获取预测掩码
        pred_mask = pred_dict["mask"]
        
        # 转换为 numpy 数组（如果是 tensor）
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        
        # 二值化（阈值 0.5）
        pred_mask_binary = (pred_mask > 0.5).astype(np.float32)
        gt_mask_binary = (gt_mask > 0.5).astype(np.float32)
        
        # 计算 IoU
        intersection = np.sum(pred_mask_binary * gt_mask_binary)
        union = np.sum(np.maximum(pred_mask_binary, gt_mask_binary))
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        
        ious.append(iou)
        
        # 累计总的交集和并集（用于计算 overall IoU）
        total_intersection += intersection
        total_union += union
        
        # 计算 Precision 和 Recall
        if np.sum(pred_mask_binary) > 0:
            precision = intersection / np.sum(pred_mask_binary)
        else:
            precision = 0.0
        
        if np.sum(gt_mask_binary) > 0:
            recall = intersection / np.sum(gt_mask_binary)
        else:
            recall = 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # 计算各项指标
    metrics = {
        "mean_iou": np.mean(ious),  # 平均 IoU
        "overall_iou": total_intersection / total_union if total_union > 0 else 0.0,  # 整体 IoU
        "precision": np.mean(precisions),  # 平均精确率
        "recall": np.mean(recalls),  # 平均召回率
        "accuracy": np.mean([iou > 0.5 for iou in ious]),  # 准确率（IoU > 0.5 的比例）
    }
    
    return metrics


def predict_model_by_single_sample(
        model: nn.Module, 
        dataset: Union[RefCOCODataset, ReasonSegDataset]
) -> Dict[str, List[Dict]]:
    """单样本预测
    
    逐个样本进行前向传播
    
    Args:
        model: PyTorch 模型
        dataset: 数据集
    
    Returns:
        结果字典，格式为 {"predictions": [{"mask": ...}, ...]}
    """
    predictions: List[Dict] = []
    model.eval()
    
    with torch.no_grad():
        for sample in dataset:
            result = forward_single_sample(model, sample)
            predictions.append(result)
    
    return {"predictions": predictions}


def predict_model_by_batch_samples(
        model: nn.Module,
        dataset: Union[RefCOCODataset, ReasonSegDataset],
        batch_size: int = 16
) -> Dict[str, List[Dict]]:
    """批量预测
    
    批量进行前向传播，提高效率
    
    Args:
        model: PyTorch 模型
        dataset: 数据集
        batch_size: 批次大小
    
    Returns:
        结果字典，格式为 {"predictions": [{"mask": ...}, ...]}
    """
    predictions: List[Dict] = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_samples = [
                dataset[j] for j in range(i, min(i + batch_size, len(dataset)))
            ]
            batch_results = forward_batch_samples(model, batch_samples)
            predictions.extend(batch_results)
    
    return {"predictions": predictions}


def forward_single_sample(model: nn.Module, example: Dict) -> Dict:
    """单样本前向传播
    
    Args:
        model: PyTorch 模型
        example: 单个样本，包含 'image' 和 'text' 等键
    
    Returns:
        预测结果字典，包含 'mask' 键
    """
    device = next(model.parameters()).device
    
    # 提取图像和文本
    image = example["image"]
    text = example.get("text", "")
    
    # 转换为 tensor 并添加 batch 维度
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    
    # 添加 batch 维度: [C, H, W] -> [1, C, H, W]
    image = image.unsqueeze(0).to(device)
    
    # 模型推理
    pred_mask = model(image, [text])  # [1, H, W]
    
    # 移除 batch 维度并转回 CPU
    pred_mask = pred_mask.squeeze(0).cpu()  # [H, W]
    
    prediction = {
        "mask": pred_mask,
        "image_id": example.get("image_id", "unknown"),
    }
    
    return prediction


def forward_batch_samples(model: nn.Module, examples: List[Dict]) -> List[Dict]:
    """批量前向传播
    
    Args:
        model: PyTorch 模型
        examples: 样本列表
    
    Returns:
        预测结果列表
    """
    device = next(model.parameters()).device
    
    # 批量整理数据
    images = []
    texts = []
    image_ids = []
    
    for example in examples:
        image = example["image"]
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        images.append(image)
        texts.append(example.get("text", ""))
        image_ids.append(example.get("image_id", "unknown"))
    
    # 堆叠为批次: [B, C, H, W]
    images_batch = torch.stack(images).to(device)
    
    # 模型推理
    pred_masks = model(images_batch, texts)  # [B, H, W]
    
    # 解析结果
    predictions: List[Dict] = []
    for i in range(len(examples)):
        pred_mask = pred_masks[i].cpu()  # [H, W]
        predictions.append({
            "mask": pred_mask,
            "image_id": image_ids[i],
        })
    
    return predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RefCOCO 数据集评估")
    parser.add_argument("--split", type=str, default="val", help="数据集划分（默认: val），可选: val, test, testA, testB")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数量，用于快速测试")
    
    args = parser.parse_args()
    
    main(split=args.split, max_samples=args.max_samples)