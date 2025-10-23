"""
LISA模型评估示例
演示如何使用eval.py对LISA模型进行RefCOCO和ReasonSeg数据集评估
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor
from PIL import Image

# 添加上级目录到路径以导入eval
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import RefCOCODataset, ReasonSegDataset, predict_model_by_single_sample, compute_metrics

# 导入LISA模型相关
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

# 设置模型下载缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_download")
os.makedirs(CACHE_DIR, exist_ok=True)

# 设置本地 SAM 权重路径
SAM_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam", "sam_vit_h_4b8939.pth")

print(f"模型缓存目录: {CACHE_DIR}")
print(f"SAM 权重路径: {SAM_CHECKPOINT_PATH}")


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


# 全局变量存储模型组件
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_CLIP_PROCESSOR = None
GLOBAL_TRANSFORM = None
GLOBAL_ARGS = None


def load_model(
    version="xinlai/LISA-13B-llama2-v1",
    precision="fp32",
    vision_tower="openai/clip-vit-large-patch14",
    sam_checkpoint=None,
    image_size=1024,
    model_max_length=512,
    use_mm_start_end=True,
    conv_type="llava_v1",
    local_rank=0,
    **kwargs
):
    """
    加载LISA模型
    
    Args:
        version: 模型版本或路径
        precision: 精度 (fp32, fp16, bf16)
        vision_tower: CLIP vision tower
        sam_checkpoint: SAM模型权重路径，默认使用 LISA/sam/sam_vit_h_4b8939.pth
        image_size: 图像尺寸
        model_max_length: 最大序列长度
        use_mm_start_end: 是否使用图像起止标记
        conv_type: 对话模板类型
        local_rank: 设备rank
    """
    global GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_CLIP_PROCESSOR, GLOBAL_TRANSFORM, GLOBAL_ARGS
    
    # 使用默认的本地 SAM 权重路径
    if sam_checkpoint is None:
        sam_checkpoint = SAM_CHECKPOINT_PATH
    
    # 检查 SAM 权重文件是否存在
    if not os.path.exists(sam_checkpoint):
        print(f"⚠️  警告: SAM 权重文件不存在: {sam_checkpoint}")
        print("   请确保 sam/sam_vit_h_4b8939.pth 文件存在")
        sam_checkpoint = None
    else:
        print(f"✓ 使用本地 SAM 权重: {sam_checkpoint}")
    
    print(f"Loading LISA model from {version}...")
    print(f"使用缓存目录: {CACHE_DIR}")
    
    # 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        version,
        cache_dir=CACHE_DIR,
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
    
    # 加载模型
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "cache_dir": CACHE_DIR,
        "vision_pretrained": sam_checkpoint  # 传递本地 SAM 权重路径
    }
    model = LISAForCausalLM.from_pretrained(
        version, 
        low_cpu_mem_usage=True, 
        vision_tower=vision_tower, 
        seg_token_idx=seg_token_idx, 
        **model_kwargs
    )
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 初始化vision模块（会使用 vision_pretrained 参数加载本地 SAM 权重）
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    
    # 将模型移到GPU
    if precision == "bf16":
        model = model.bfloat16().cuda()
    elif precision == "fp16":
        model = model.half().cuda()
    else:
        model = model.float().cuda()
    
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=local_rank)
    
    model.eval()
    
    # 创建图像处理器
    clip_image_processor = CLIPImageProcessor.from_pretrained(
        model.config.vision_tower,
        cache_dir=CACHE_DIR
    )
    transform = ResizeLongestSide(image_size)
    
    # 保存到全局变量
    GLOBAL_MODEL = model
    GLOBAL_TOKENIZER = tokenizer
    GLOBAL_CLIP_PROCESSOR = clip_image_processor
    GLOBAL_TRANSFORM = transform
    
    # 保存参数
    class Args:
        pass
    args = Args()
    args.precision = precision
    args.use_mm_start_end = use_mm_start_end
    args.conv_type = conv_type
    args.seg_token_idx = seg_token_idx
    args.image_size = image_size
    GLOBAL_ARGS = args
    
    print("Model loaded successfully!")
    return model


def forward_single_sample(model, example):
    """
    LISA模型单样本推理
    
    对于ReasonSeg任务，输出分割掩码
    对于RefCOCO任务，输出文本回答
    """
    global GLOBAL_TOKENIZER, GLOBAL_CLIP_PROCESSOR, GLOBAL_TRANSFORM, GLOBAL_ARGS
    
    # 获取输入
    image_pil = example['image']  # PIL Image
    text = example['text']  # str
    
    # 将PIL Image转换为numpy
    image_np = np.array(image_pil)
    if image_np.ndim == 2:  # 灰度图
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
    original_size_list = [image_np.shape[:2]]
    
    # 准备prompt
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + text
    if GLOBAL_ARGS.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    
    conv = conversation_lib.conv_templates[GLOBAL_ARGS.conv_type].copy()
    conv.messages = []
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()
    
    # 处理图像 - CLIP
    image_clip = (
        GLOBAL_CLIP_PROCESSOR.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
    )
    if GLOBAL_ARGS.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif GLOBAL_ARGS.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()
    
    # 处理图像 - SAM
    image_sam = GLOBAL_TRANSFORM.apply_image(image_np)
    resize_list = [image_sam.shape[:2]]
    
    image_sam = (
        preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if GLOBAL_ARGS.precision == "bf16":
        image_sam = image_sam.bfloat16()
    elif GLOBAL_ARGS.precision == "fp16":
        image_sam = image_sam.half()
    else:
        image_sam = image_sam.float()
    
    # Tokenize
    input_ids = tokenizer_image_token(prompt, GLOBAL_TOKENIZER, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    
    # 模型推理
    with torch.no_grad():
        output_ids, pred_masks = model.evaluate(
            image_clip,
            image_sam,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=GLOBAL_TOKENIZER,
        )
    
    # 解码文本输出
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_output = GLOBAL_TOKENIZER.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    
    # 处理掩码输出
    pred_mask = None
    if pred_masks and len(pred_masks) > 0:
        for mask in pred_masks:
            if mask.shape[0] > 0:
                pred_mask = mask.detach().cpu().numpy()[0]
                pred_mask = (pred_mask > 0).astype(np.uint8)
                break
    
    # 根据任务类型返回结果
    result = {
        'answer': text_output,  # RefCOCO需要
    }
    
    if pred_mask is not None:
        result['mask'] = pred_mask  # ReasonSeg需要
    
    return result


def forward_batch_samples(model, examples):
    """
    批量推理 - LISA不太支持批量，所以逐个处理
    """
    predictions = []
    for example in examples:
        pred = forward_single_sample(model, example)
        predictions.append(pred)
    return predictions


def main():
    """
    主函数 - 演示如何使用
    """
    print("="*50)
    print("LISA Model Evaluation Example")
    print("="*50)
    print(f"\n💡 模型配置信息:")
    print(f"   - 模型缓存目录: {CACHE_DIR}")
    print(f"   - SAM 权重路径: {SAM_CHECKPOINT_PATH}")
    if os.path.exists(SAM_CHECKPOINT_PATH):
        print(f"   ✓ SAM 权重文件已找到")
    else:
        print(f"   ✗ SAM 权重文件不存在，请检查路径")
    print()
    
    # ============= 示例1: ReasonSeg数据集评估 =============
    print("\n[Example 1] ReasonSeg Dataset Evaluation")
    print("-"*50)
    
    # 1. 加载模型（如果你有本地权重，修改version参数）
    # 注意：这里假设你已经下载了LISA模型权重
    # 如果没有，请先下载或指定正确的模型路径
    try:
        model = load_model(
            version="xinlai/LISA-13B-llama2-v1",  # 或本地路径
            precision="fp32",  # 根据显存选择: fp32, fp16, bf16
            vision_tower="openai/clip-vit-large-patch14",
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("提示: 请确保已安装LISA并下载了模型权重")
        print("或者修改version参数指向本地模型路径")
        return
    
    # 2. 加载ReasonSeg数据集
    # 假设你的ReasonSeg数据在上级目录的ReasonSeg/val文件夹
    reasonseg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ReasonSeg", "val")
    
    if os.path.exists(reasonseg_path):
        print(f"Loading ReasonSeg dataset from: {reasonseg_path}")
        dataset = ReasonSegDataset(image_folder=reasonseg_path)
        print(f"Dataset size: {len(dataset)}")
        
        # 3. 运行预测（这里只测试前5个样本作为演示）
        print("\nRunning predictions on first 5 samples...")
        
        # 创建小的测试集
        class SmallDataset:
            def __init__(self, dataset, num_samples=5):
                self.dataset = dataset
                self.num_samples = min(num_samples, len(dataset))
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                return self.dataset[idx]
        
        small_dataset = SmallDataset(dataset, num_samples=5)
        
        # 运行预测
        results = predict_model_by_single_sample(model, small_dataset)
        
        # 4. 计算指标
        print("\nComputing metrics...")
        metrics = compute_metrics(dataset, results)
        
        print("\n" + "="*50)
        print("ReasonSeg Evaluation Results:")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    else:
        print(f"ReasonSeg dataset not found at: {reasonseg_path}")
        print("请确保数据集路径正确")
    
    
    # ============= 示例2: RefCOCO数据集评估 =============
    print("\n\n[Example 2] RefCOCO Dataset Evaluation")
    print("-"*50)
    
    # 假设你有RefCOCO的JSONL文件
    refcoco_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "refcoco", "refcoco_val.jsonl")
    
    if os.path.exists(refcoco_path):
        print(f"Loading RefCOCO dataset from: {refcoco_path}")
        
        # 对于RefCOCO，修改prompt模板
        refcoco_dataset = RefCOCODataset(
            jsonl_path=refcoco_path,
            prompt_template="Can you segment the {} in this image?"
        )
        print(f"Dataset size: {len(refcoco_dataset)}")
        
        # 运行预测（测试前5个）
        small_refcoco = SmallDataset(refcoco_dataset, num_samples=5)
        results = predict_model_by_single_sample(model, small_refcoco)
        
        # 计算指标
        metrics = compute_metrics(refcoco_dataset, results)
        
        print("\n" + "="*50)
        print("RefCOCO Evaluation Results:")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    else:
        print(f"RefCOCO dataset not found at: {refcoco_path}")
        print("如果需要评估RefCOCO，请准备数据集")
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)


if __name__ == "__main__":
    main()

