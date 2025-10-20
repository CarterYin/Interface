"""Lightweight dataset and evaluation utilities for demonstration purposes."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    """Container describing one dataset sample."""

    sample_id: int
    feature: Sequence[float]
    label: int
    metadata: Optional[Dict[str, Any]] = None


class RefCOCODataset(Dataset):
    """Minimal in-memory dataset mimicking RefCOCO style annotations."""

    def __init__(
        self,
        samples: Optional[Iterable[Sample]] = None,
    ) -> None:
        if samples is None:
            samples = self._generate_default_samples()
        self._samples: List[Sample] = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        return {
            "id": sample.sample_id,
            "feature": list(sample.feature),
            "label": sample.label,
            "metadata": sample.metadata or {},
        }

    def __iter__(self):
        for index in range(len(self._samples)):
            yield self[index]

    @staticmethod
    def _generate_default_samples() -> List[Sample]:
        default_samples: List[Sample] = []
        for idx in range(20):
            feature_vector = [(idx % 3) / 2.0, (idx % 5) / 4.0, (idx % 7) / 6.0]
            label = int(mean(feature_vector) >= 0.5)
            default_samples.append(
                Sample(
                    sample_id=idx,
                    feature=feature_vector,
                    label=label,
                    metadata={"split": "train" if idx < 15 else "val"},
                )
            )
        return default_samples


class ReasonSegDataset(Dataset):
    """Simple dataset providing segmentation-style ground-truth masks."""

    def __init__(self, masks: Optional[List[List[List[int]]]] = None) -> None:
        if masks is None:
            masks = self._generate_default_masks()
        self._masks = masks

    def __len__(self) -> int:
        return len(self._masks)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        mask = self._masks[index]
        return {
            "id": index,
            "mask": mask,
            "area": sum(sum(row) for row in mask),
        }

    def __iter__(self):
        for index in range(len(self._masks)):
            yield self[index]

    @staticmethod
    def _generate_default_masks() -> List[List[List[int]]]:
        masks: List[List[List[int]]] = []
        for size in range(3, 8):
            square_mask = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
            masks.append(square_mask)
        return masks


def main():
    dataset = RefCOCODataset()
    model = load_model()  # Load or initialize your model here
    # first conduct prediction
    single_sample_results = predict_model_by_single_sample(model, dataset)
    batch_sample_results = (
        predict_model_by_batch_samples(model, dataset, batch_size=16)
    )
    # then compute metrics
    metrics_by_single_samples = compute_metrics(dataset, single_sample_results)
    metrics_by_batch_samples = compute_metrics(dataset, batch_sample_results)


def load_model(**kwargs):
    return SimpleThresholdModel(**kwargs)


def compute_metrics(dataset: RefCOCODataset, results: Dict) -> Dict:
    ground_truth: Dict[int, int] = {
        sample["id"]: sample["label"] for sample in dataset
    }
    predictions = results.get("predictions", [])
    correct = 0
    squared_error_sum = 0.0
    evaluated_items = 0
    for prediction in predictions:
        sample_id = prediction.get("id")
        if sample_id is None or sample_id not in ground_truth:
            continue
        evaluated_items += 1
        gt_label = ground_truth[sample_id]
        pred_label = prediction.get("predicted_label")
        score = prediction.get("score", float(pred_label or 0))
        if pred_label == gt_label:
            correct += 1
        squared_error_sum += float(score - gt_label) ** 2
    accuracy = correct / evaluated_items if evaluated_items else 0.0
    mse = squared_error_sum / evaluated_items if evaluated_items else 0.0
    return {
        "evaluated_samples": evaluated_items,
        "accuracy": accuracy,
        "mse": mse,
    }


def predict_model_by_single_sample(
        model, dataset: RefCOCODataset) -> Dict[str, List[Dict]]:
    predictions: List[Dict] = []
    for sample in dataset:
        result = forward_single_sample(model, sample)
        predictions.append(result)
    return {"predictions": predictions}


def predict_model_by_batch_samples(
        model,
        dataset: RefCOCODataset,
        batch_size: int) -> Dict[str, List[Dict]]:
    predictions: List[Dict] = []
    for i in range(0, len(dataset), batch_size):
        batch_samples = [
            dataset[j] for j in range(i, min(i + batch_size, len(dataset)))
        ]
        batch_results = forward_batch_samples(model, batch_samples)
        predictions.extend(batch_results)
    return {"predictions": predictions}


def forward_single_sample(model, example: Dict) -> Dict:
    return model.predict_single(example)


def forward_batch_samples(model, examples: List[Dict]) -> List[Dict]:
    return model.predict_batch(examples)


class SimpleThresholdModel:
    """Tiny deterministic model used for illustrating the workflow."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def predict_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        feature = sample.get("feature", [])
        score = mean(feature) if feature else 0.0
        predicted_label = int(score >= self.threshold)
        return {
            "id": sample.get("id"),
            "predicted_label": predicted_label,
            "score": score,
        }

    def predict_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict_single(sample) for sample in samples]
