import json
import re
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RefCOCODataset(Dataset):
    """RefCOCO dataset for referring expression grounding evaluation.
    
    Expected data format: JSONL file with each line containing:
    {
        "image": "path/to/image.jpg",
        "sent": "referring expression text",
        "bbox": [x1, y1, x2, y2],  # in pixel coordinates
        "width": image_width,
        "height": image_height
    }
    """
    
    def __init__(self, jsonl_path: str, prompt_template: str = None):
        """
        Args:
            jsonl_path: Path to JSONL annotation file
            prompt_template: Optional prompt template with {} placeholder for text.
                           Default: 'Please provide the bounding box coordinate of 
                           the region this sentence describes: {}'
        """
        self.datas = open(jsonl_path).readlines()
        if prompt_template is None:
            self.prompt_template = (
                'Please provide the bounding box coordinate of the region '
                'this sentence describes: {}'
            )
        else:
            self.prompt_template = prompt_template
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        image_path = data['image']
        text = data['sent']
        bbox = data['bbox']  # [x1, y1, x2, y2]
        w, h = data['width'], data['height']
        
        image = Image.open(image_path).convert('RGB')
        
        return {
            'image': image,
            'text': self.prompt_template.format(text),
            'gt_bbox': bbox,
            'hw': (h, w)
        }


class ReasonSegDataset(Dataset):
    """ReasonSeg dataset for reasoning segmentation evaluation.
    
    Expected data structure: 
    - image_folder containing .jpg images and corresponding .json annotation files
    - Each .json file contains segmentation mask and text descriptions
    """
    
    def __init__(self, image_folder: str):
        """
        Args:
            image_folder: Path to folder containing images and JSON annotations
        """
        import glob
        import os
        self.image_folder = image_folder
        self.images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        import cv2
        image_path = self.images[idx]
        json_path = image_path.replace(".jpg", ".json")
        
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        # Load mask and text from JSON
        with open(json_path, "r") as f:
            anno = json.load(f)
        
        # Parse annotations
        shapes = anno.get("shapes", [])
        text = anno.get("text", "")
        
        # Generate ground truth mask
        mask = np.zeros((height, width), dtype=np.uint8)
        for shape in shapes:
            points = shape["points"]
            label = shape.get("label", "")
            if "ignore" in label.lower() or "flag" in label.lower():
                continue
            cv2.fillPoly(mask, np.array([points], dtype=np.int32), 1)
        
        return {
            'image': image,
            'text': text,
            'gt_mask': mask,
            'hw': (height, width)
        }


def main():
    # Example usage
    dataset = RefCOCODataset("path/to/refcoco_val.jsonl")
    model = load_model()  # Load or initialize your model here
    # first conduct prediction
    single_sample_results = predict_model_by_single_sample(model, dataset)
    batch_sample_results = (
        predict_model_by_batch_samples(model, dataset, batch_size=16)
    )
    # then compute metrics
    metrics_by_single_samples = compute_metrics(dataset, single_sample_results)
    metrics_by_batch_samples = compute_metrics(dataset, batch_sample_results)
    print("Metrics (single):", metrics_by_single_samples)
    print("Metrics (batch):", metrics_by_batch_samples)


def load_model(**kwargs):
    """Load your model here.
    
    Example:
        from your_model import YourModel
        model = YourModel.from_pretrained("model_path")
        return model
    """
    model = None
    # Implement model loading logic here
    return model


def compute_metrics(dataset, results: Dict) -> Dict:
    """Compute evaluation metrics based on dataset type and predictions.
    
    Args:
        dataset: RefCOCODataset or ReasonSegDataset instance
        results: Dict with 'predictions' key containing list of prediction dicts
        
    Returns:
        Dict containing computed metrics
    """
    predictions = results['predictions']
    metrics = {}
    
    if isinstance(dataset, RefCOCODataset):
        # RefCOCO evaluation: bounding box IoU
        correct = 0
        total = 0
        iou_list = []
        pattern = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')
        
        for idx, pred in enumerate(predictions):
            gt_sample = dataset[idx]
            gt_bbox = gt_sample['gt_bbox']  # [x1, y1, x2, y2]
            h, w = gt_sample['hw']
            
            # Parse predicted bbox from answer text
            pred_text = pred.get('answer', pred.get('text', ''))
            bbox_match = pattern.findall(pred_text)
            
            if bbox_match and len(bbox_match[0]) == 4:
                try:
                    pred_bbox = [float(x) for x in bbox_match[0]]
                except:
                    pred_bbox = [0., 0., 0., 0.]
            else:
                pred_bbox = [0., 0., 0., 0.]
            
            # Convert to tensors
            pred_bbox_tensor = torch.tensor(pred_bbox, dtype=torch.float32).view(-1, 4)
            gt_bbox_tensor = torch.tensor(gt_bbox, dtype=torch.float32).view(-1, 4)
            
            # If predicted bbox is normalized [0, 1], denormalize it
            if pred_bbox_tensor.sum() < 4:
                pred_bbox_tensor[:, 0::2] *= w
                pred_bbox_tensor[:, 1::2] *= h
            
            # Compute IoU using box_iou formula
            def box_area(boxes):
                return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
            area1 = box_area(pred_bbox_tensor)
            area2 = box_area(gt_bbox_tensor)
            lt = torch.max(pred_bbox_tensor[:, None, :2], gt_bbox_tensor[:, :2])
            rb = torch.min(pred_bbox_tensor[:, None, 2:], gt_bbox_tensor[:, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[:, :, 0] * wh[:, :, 1]
            union = area1[:, None] + area2 - inter
            iou = inter / union
            
            iou_value = iou.item() if isinstance(iou, torch.Tensor) else iou
            iou_list.append(iou_value)
            
            total += 1
            if iou_value >= 0.5:
                correct += 1
        
        metrics = {
            'Precision@0.5': correct / total if total > 0 else 0.0,
            'mean_iou': np.mean(iou_list) if iou_list else 0.0,
            'total_samples': total
        }
        
    elif isinstance(dataset, ReasonSegDataset):
        # ReasonSeg evaluation: mask IoU
        iou_list = []
        
        for idx, pred in enumerate(predictions):
            gt_sample = dataset[idx]
            gt_mask = gt_sample['gt_mask']  # numpy array (H, W)
            
            # Get predicted mask
            pred_mask = pred.get('mask', None)
            
            if pred_mask is None:
                iou_list.append(0.0)
                continue
            
            # Ensure mask is numpy array
            if isinstance(pred_mask, torch.Tensor):
                pred_mask = pred_mask.cpu().numpy()
            
            # Compute IoU
            intersection = np.logical_and(gt_mask > 0, pred_mask > 0).sum()
            union = np.logical_or(gt_mask > 0, pred_mask > 0).sum()
            
            iou = intersection / union if union > 0 else 0.0
            iou_list.append(iou)
        
        # Compute precision at different thresholds
        iou_array = np.array(iou_list)
        metrics = {
            'P@0.5': np.sum(iou_array > 0.5) / len(iou_array) if len(iou_array) > 0 else 0.0,
            'P@0.6': np.sum(iou_array > 0.6) / len(iou_array) if len(iou_array) > 0 else 0.0,
            'P@0.7': np.sum(iou_array > 0.7) / len(iou_array) if len(iou_array) > 0 else 0.0,
            'P@0.8': np.sum(iou_array > 0.8) / len(iou_array) if len(iou_array) > 0 else 0.0,
            'P@0.9': np.sum(iou_array > 0.9) / len(iou_array) if len(iou_array) > 0 else 0.0,
            'mean_iou': np.mean(iou_array) if len(iou_array) > 0 else 0.0,
            'total_samples': len(iou_array)
        }
    else:
        raise ValueError(f"Unknown dataset type: {type(dataset)}")
    
    return metrics


def predict_model_by_single_sample(
        model, dataset) -> Dict[str, List[Dict]]:
    """Run prediction on dataset sample by sample.
    
    Args:
        model: Your loaded model
        dataset: RefCOCODataset or ReasonSegDataset instance
        
    Returns:
        Dict with 'predictions' key containing list of prediction dicts
    """
    predictions: List[Dict] = []
    for sample in dataset:
        result = forward_single_sample(model, sample)
        predictions.append(result)
    return {"predictions": predictions}


def predict_model_by_batch_samples(
        model,
        dataset,
        batch_size: int) -> Dict[str, List[Dict]]:
    """Run prediction on dataset in batches.
    
    Args:
        model: Your loaded model
        dataset: RefCOCODataset or ReasonSegDataset instance
        batch_size: Number of samples per batch
        
    Returns:
        Dict with 'predictions' key containing list of prediction dicts
    """
    predictions: List[Dict] = []
    for i in range(0, len(dataset), batch_size):
        batch_samples = [
            dataset[j] for j in range(i, min(i + batch_size, len(dataset)))
        ]
        batch_results = forward_batch_samples(model, batch_samples)
        predictions.extend(batch_results)
    return {"predictions": predictions}


def forward_single_sample(model, example: Dict) -> Dict:
    """Forward pass for a single sample.
    
    Args:
        model: Your loaded model
        example: Dict containing 'image', 'text', and ground truth data
        
    Returns:
        Dict with prediction results. For RefCOCO, should contain 'answer' with
        bbox coordinates. For ReasonSeg, should contain 'mask' as numpy array.
        
    Example for RefCOCO:
        image = example['image']  # PIL Image
        text = example['text']    # str
        # Your model inference code here
        output = model.generate(image, text)
        return {'answer': output}  # output should contain bbox like "[x1,y1,x2,y2]"
        
    Example for ReasonSeg:
        image = example['image']  # PIL Image  
        text = example['text']    # str
        # Your model inference code here
        mask = model.segment(image, text)  # numpy array (H, W)
        return {'mask': mask}
    """
    prediction = {}
    # Implement single sample forward logic here
    # This is where you call your model
    return prediction


def forward_batch_samples(model, examples: List[Dict]) -> List[Dict]:
    """Forward pass for a batch of samples.
    
    Args:
        model: Your loaded model
        examples: List of dicts, each containing 'image', 'text', and ground truth
        
    Returns:
        List of dicts with prediction results
        
    Example:
        images = [ex['image'] for ex in examples]
        texts = [ex['text'] for ex in examples]
        # Your batch inference code here
        outputs = model.batch_generate(images, texts)
        return [{'answer': out} for out in outputs]
    """
    predictions: List[Dict] = []
    # Implement batch forward logic here
    # You can also just call forward_single_sample for each example
    for example in examples:
        pred = forward_single_sample(model, example)
        predictions.append(pred)
    return predictions
