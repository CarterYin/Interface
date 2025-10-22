import os
from typing import Dict, List, Union, Optional
import json

# ============================================================
# 设置缓存路径到 E 盘（避免占用 C 盘空间）
# 如果不需要，可以注释掉以下代码
# ============================================================
CACHE_DIR = "E:/model_cache"  # 修改此路径以更改缓存位置
os.makedirs(f"{CACHE_DIR}/huggingface", exist_ok=True)
os.makedirs(f"{CACHE_DIR}/huggingface/hub", exist_ok=True)
os.makedirs(f"{CACHE_DIR}/huggingface/transformers", exist_ok=True)
os.makedirs(f"{CACHE_DIR}/torch", exist_ok=True)

os.environ['HF_HOME'] = f"{CACHE_DIR}/huggingface"
os.environ['HUGGINGFACE_HUB_CACHE'] = f"{CACHE_DIR}/huggingface/hub"
os.environ['TRANSFORMERS_CACHE'] = f"{CACHE_DIR}/huggingface/transformers"
os.environ['TORCH_HOME'] = f"{CACHE_DIR}/torch"

print(f"[INFO] 模型缓存路径已设置到: {CACHE_DIR}")
# ============================================================

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
    数据集包含图像和对应的多边形标注，标注包括文本描述和分割区域
    
    Args:
        data_root: 数据根目录（包含 train/val/test 子目录）
        split: 数据集划分，如 'train', 'val', 'test'
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
        
        # 构建数据目录路径
        self.split_dir = os.path.join(data_root, split)
        
        if not os.path.exists(self.split_dir):
            raise ValueError(f"数据目录不存在: {self.split_dir}")
        
        # 获取所有图像文件
        print(f"正在加载 ReasonSeg 数据集: split={split}, 目录={self.split_dir}")
        all_files = os.listdir(self.split_dir)
        
        # 只保留 .jpg 文件，并确保对应的 .json 文件存在
        self.image_files = []
        for f in all_files:
            if f.endswith('.jpg'):
                json_file = f.replace('.jpg', '.json')
                if json_file in all_files:
                    self.image_files.append(f)
        
        self.image_files = sorted(self.image_files)
        print(f"数据集加载完成，共 {len(self.image_files)} 个样本")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_files)
    
    def _polygon_to_mask(self, points: List[List[float]], img_width: int, img_height: int) -> np.ndarray:
        """将多边形坐标转换为二值掩码
        
        Args:
            points: 多边形顶点坐标列表 [[x1, y1], [x2, y2], ...]
            img_width: 图像宽度
            img_height: 图像高度
        
        Returns:
            mask: 二值掩码数组 [H, W]
        """
        from PIL import ImageDraw
        
        # 创建空白掩码
        mask = Image.new('L', (img_width, img_height), 0)
        draw = ImageDraw.Draw(mask)
        
        # 将points转换为扁平化的坐标列表 [x1, y1, x2, y2, ...]
        polygon_coords = []
        for point in points:
            polygon_coords.extend([float(point[0]), float(point[1])])
        
        # 绘制多边形
        if len(polygon_coords) >= 6:  # 至少需要3个点（6个坐标值）
            draw.polygon(polygon_coords, fill=255)
        
        return np.array(mask)
    
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
        # 获取文件路径
        image_file = self.image_files[idx]
        json_file = image_file.replace('.jpg', '.json')
        
        image_path = os.path.join(self.split_dir, image_file)
        json_path = os.path.join(self.split_dir, json_file)
        
        # 读取图像
        image = Image.open(image_path)
        orig_width, orig_height = image.size
        
        # 确保是 RGB 模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 读取标注
        with open(json_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        # 提取文本描述
        text_list = annotation.get('text', [''])
        if isinstance(text_list, list) and len(text_list) > 0:
            text = text_list[0]  # 使用第一个文本描述
        else:
            text = str(text_list)
        
        # 处理多边形标注，生成掩码
        shapes = annotation.get('shapes', [])
        
        # 创建空白掩码
        combined_mask = np.zeros((orig_height, orig_width), dtype=np.float32)
        
        # 将所有标记为 "target" 的多边形合并到掩码中
        for shape in shapes:
            if shape.get('label') == 'target':
                points = shape.get('points', [])
                if len(points) >= 3:  # 至少需要3个点才能构成多边形
                    poly_mask = self._polygon_to_mask(points, orig_width, orig_height)
                    combined_mask = np.maximum(combined_mask, poly_mask.astype(np.float32) / 255.0)
        
        # 调整图像和掩码大小
        image = image.resize((self.image_size, self.image_size))
        mask = Image.fromarray((combined_mask * 255).astype(np.uint8))
        mask = mask.resize((self.image_size, self.image_size))
        
        # 转换为 tensor
        image = torch.from_numpy(np.array(image)).float()
        # [H, W, C] -> [C, H, W]
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1)
        # 归一化到 [0, 1]
        if image.max() > 1:
            image = image / 255.0
        
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        
        # 提取边界框（如果需要）
        # 从掩码计算边界框
        mask_np = mask.numpy()
        ys, xs = np.where(mask_np > 0.5)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        else:
            bbox = None
        
        # 使用文件名作为 image_id
        image_id = image_file.replace('.jpg', '')
        
        return {
            "image_id": image_id,
            "image": image,
            "text": text,
            "mask": mask,
            "bbox": bbox,
        }


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


class LISAModelAdapter(nn.Module):
    """LISA 模型适配器
    
    将 LISA 模型包装成与 eval.py 兼容的接口
    
    LISA 需要特殊的输入格式：
    - 图像预处理（CLIP + SAM）
    - 文本提示格式化
    - 特殊的 token 处理
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        clip_image_processor,
        transform,
        image_size=1024,
        precision="bf16"
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.clip_image_processor = clip_image_processor
        self.transform = transform
        self.image_size = image_size
        self.precision = precision
        
        # 用于图像标准化的参数
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).cuda()
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).cuda()
    
    def preprocess_image(self, image_tensor: torch.Tensor) -> tuple:
        """预处理图像用于 LISA
        
        Args:
            image_tensor: [C, H, W] 图像，值域 [0, 1]
        
        Returns:
            tuple: (clip_image, resize_image, original_size, resized_size)
        """
        # 转换回 PIL Image 或 numpy 进行处理
        image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # CLIP 图像处理
        clip_image = self.clip_image_processor.preprocess(
            image_np, return_tensors="pt"
        )["pixel_values"][0]
        
        # SAM 图像处理
        original_size = image_np.shape[:2]
        resize_image = self.transform.apply_image(image_np)
        resized_size = resize_image.shape[:2]
        
        # 转换为 torch tensor 并标准化
        resize_image = torch.from_numpy(resize_image).permute(2, 0, 1).float()
        resize_image = (resize_image - self.pixel_mean) / self.pixel_std
        
        # Pad 到正方形
        h, w = resize_image.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        resize_image = torch.nn.functional.pad(resize_image, (0, padw, 0, padh))
        
        return clip_image, resize_image, original_size, resized_size
    
    def forward(self, images: torch.Tensor, texts: Optional[List[str]] = None) -> torch.Tensor:
        """
        适配接口：将 eval.py 的标准输入转换为 LISA 所需格式
        
        Args:
            images: [B, 3, H, W] 图像张量，值域 [0, 1]
            texts: 长度为 B 的文本描述列表
        
        Returns:
            masks: [B, H, W] 预测掩码，值域 [0, 1]
        """
        from model.llava.mm_utils import tokenizer_image_token
        from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from model.llava import conversation as conversation_lib
        
        batch_size = images.shape[0]
        device = images.device
        
        # 处理每个样本
        output_masks = []
        
        for i in range(batch_size):
            image = images[i]  # [3, H, W]
            text = texts[i] if texts else "Please segment the object."
            
            # 预处理图像
            clip_image, resize_image, original_size, resized_size = self.preprocess_image(image)
            clip_image = clip_image.unsqueeze(0).to(device)
            resize_image = resize_image.unsqueeze(0).to(device)
            
            # 构建对话提示
            # LISA 使用 "[SEG]" token 来触发分割
            prompt = f"{text}. [SEG]"
            
            # 添加图像 token
            prompt_with_image = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            
            # Tokenize
            input_ids = tokenizer_image_token(
                prompt_with_image,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt"
            ).unsqueeze(0).to(device)
            
            # 准备模型输入
            with torch.no_grad():
                output_ids, pred_masks = self.model.evaluate(
                    clip_image,
                    resize_image,
                    input_ids,
                    max_new_tokens=512,
                    tokenizer=self.tokenizer,
                )
            
            # 处理预测掩码
            # pred_masks 通常是 [1, 1, H, W] 格式
            if pred_masks is not None and len(pred_masks) > 0:
                pred_mask = pred_masks[0]  # 取第一个掩码
                
                # 调整到原始评估图像大小（eval.py 的 image_size）
                eval_size = image.shape[-2:]  # [H, W]
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=eval_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                
                # 确保值域在 [0, 1]
                pred_mask = torch.sigmoid(pred_mask) if pred_mask.max() > 1 else pred_mask
            else:
                # 如果没有预测结果，返回零掩码
                eval_size = image.shape[-2:]
                pred_mask = torch.zeros(eval_size, device=device)
            
            output_masks.append(pred_mask)
        
        # 堆叠所有掩码
        masks = torch.stack(output_masks)  # [B, H, W]
        
        return masks


def main(
    dataset_type: str = "refcoco",
    split: str = "val",
    data_root: str = None,
    max_samples: int = None
):
    """主函数示例
    
    Args:
        dataset_type: 数据集类型，可选: 'refcoco', 'reasonseg'
        split: 数据集划分
            - RefCOCO: 'val', 'test', 'testA', 'testB'
            - ReasonSeg: 'train', 'val', 'test'
        data_root: ReasonSeg 数据集的根目录（仅当 dataset_type='reasonseg' 时需要）
        max_samples: 最大样本数量，用于快速测试（None 表示使用全部数据）
    """
    # 创建数据集
    if dataset_type.lower() == "refcoco":
        print("=== RefCOCO 数据集评估 ===")
        dataset = RefCOCODataset(split=split)
    elif dataset_type.lower() == "reasonseg":
        print("=== ReasonSeg 数据集评估 ===")
        if data_root is None:
            # 默认使用当前目录下的 ReasonSeg 文件夹
            data_root = "ReasonSeg"
        dataset = ReasonSegDataset(data_root=data_root, split=split)
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}，可选: 'refcoco', 'reasonseg'")
    
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
    """加载或初始化模型（LISA 版本）
    
    Args:
        version: LISA 模型版本（默认: "xinlai/LISA-7B-v1"）
        precision: 精度 (fp32/bf16/fp16)（默认: "bf16"）
        vision_tower: Vision tower 模型（默认: "openai/clip-vit-large-patch14"）
        sam_path: SAM 权重路径（默认: "./sam/sam_vit_h_4b8939.pth"）
        load_in_8bit: 是否使用 8bit 量化
        load_in_4bit: 是否使用 4bit 量化
        device: 运行设备（可选）
    
    Returns:
        model: LISA 模型的适配器包装
    """
    from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
    from model.LISA import LISAForCausalLM
    from model.segment_anything.utils.transforms import ResizeLongestSide
    from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    
    # 参数配置
    version = kwargs.get("version", "xinlai/LISA-7B-v1")
    precision = kwargs.get("precision", "bf16")
    vision_tower = kwargs.get("vision_tower", "openai/clip-vit-large-patch14")
    sam_path = kwargs.get("sam_path", "./sam/sam_vit_h_4b8939.pth")
    load_in_8bit = kwargs.get("load_in_8bit", False)
    load_in_4bit = kwargs.get("load_in_4bit", False)
    image_size = kwargs.get("image_size", 1024)
    model_max_length = kwargs.get("model_max_length", 512)
    
    print(f"加载 LISA 模型: {version}")
    print(f"精度: {precision}, Vision Tower: {vision_tower}")
    print(f"SAM 权重路径: {sam_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        version,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    # 设置精度
    torch_dtype = torch.float32
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.half
    
    # 量化配置
    model_kwargs = {"torch_dtype": torch_dtype}
    if load_in_4bit:
        model_kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif load_in_8bit:
        model_kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        })
    
    # 加载模型
    print("正在加载 LISA 模型...")
    model = LISAForCausalLM.from_pretrained(
        version,
        low_cpu_mem_usage=True,
        vision_tower=vision_tower,
        seg_token_idx=seg_token_idx,
        **model_kwargs
    )
    
    # 配置 token IDs
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 初始化 vision modules
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower_module = model.get_model().get_vision_tower()
    vision_tower_module.to(dtype=torch_dtype)
    
    # 设置设备和精度
    if precision == "bf16":
        model = model.bfloat16().cuda()
    elif precision == "fp16" and (not load_in_4bit) and (not load_in_8bit):
        model = model.half().cuda()
    elif precision == "fp32":
        model = model.float().cuda()
    else:
        model = model.cuda()
    
    vision_tower_module = model.get_model().get_vision_tower()
    vision_tower_module.to(device="cuda")
    
    # 创建图像处理器
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(image_size)
    
    model.eval()
    
    print("✓ LISA 模型加载完成")
    
    # 使用适配器包装模型
    lisa_adapter = LISAModelAdapter(
        model=model,
        tokenizer=tokenizer,
        clip_image_processor=clip_image_processor,
        transform=transform,
        image_size=image_size,
        precision=precision
    )
    
    return lisa_adapter


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
    
    parser = argparse.ArgumentParser(description="RefCOCO/ReasonSeg 数据集评估")
    parser.add_argument(
        "--dataset",
        type=str,
        default="reasonseg",
        choices=["refcoco", "reasonseg"],
        help="数据集类型（默认: reasonseg）"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="数据集划分（默认: val）。RefCOCO: val/test/testA/testB, ReasonSeg: train/val/test"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="ReasonSeg",
        help="ReasonSeg 数据集根目录（默认: ReasonSeg）"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数量，用于快速测试（默认: None，使用全部数据）"
    )
    
    args = parser.parse_args()
    
    main(
        dataset_type=args.dataset,
        split=args.split,
        data_root=args.data_root,
        max_samples=args.max_samples
    )