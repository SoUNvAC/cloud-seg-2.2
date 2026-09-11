"""Boundary F1 metric for cloud segmentation."""

from typing import Dict, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from scipy.ndimage import binary_dilation, binary_erosion


def _boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return np.logical_xor(
        mask, binary_erosion(mask, structure=structure, border_value=0)
    )


@METRICS.register_module()
class BoundaryF1Metric(BaseMetric):
    """Macro class Boundary F1 with a fixed pixel tolerance."""

    default_prefix = "boundary"

    def __init__(self, tolerance: int = 3, ignore_index: int = 255, **kwargs) -> None:
        super().__init__(**kwargs)
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = tolerance
        self.ignore_index = ignore_index

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta["classes"])
        structure = np.ones((3, 3), dtype=bool)
        for sample in data_samples:
            pred = sample["pred_sem_seg"]["data"].squeeze().cpu().numpy()
            target = sample["gt_sem_seg"]["data"].squeeze().cpu().numpy()
            valid = target != self.ignore_index
            counts = np.zeros((num_classes, 4), dtype=np.float64)
            for class_id in range(num_classes):
                pred_boundary = _boundary((pred == class_id) & valid)
                target_boundary = _boundary((target == class_id) & valid)
                if self.tolerance:
                    pred_match_area = binary_dilation(
                        pred_boundary, structure=structure, iterations=self.tolerance
                    )
                    target_match_area = binary_dilation(
                        target_boundary, structure=structure, iterations=self.tolerance
                    )
                else:
                    pred_match_area = pred_boundary
                    target_match_area = target_boundary
                counts[class_id] = (
                    np.logical_and(pred_boundary, target_match_area).sum(),
                    pred_boundary.sum(),
                    np.logical_and(target_boundary, pred_match_area).sum(),
                    target_boundary.sum(),
                )
            self.results.append(counts)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        counts = np.stack(results).sum(axis=0)
        class_names = self.dataset_meta["classes"]
        scores = []
        metrics: Dict[str, float] = {}
        for class_id, class_name in enumerate(class_names):
            matched_pred, pred_total, matched_target, target_total = counts[class_id]
            if pred_total == 0 and target_total == 0:
                score = np.nan
            else:
                precision = matched_pred / max(pred_total, 1.0)
                recall = matched_target / max(target_total, 1.0)
                score = 2.0 * precision * recall / max(precision + recall, 1e-12)
            metrics[f"{class_name}_F1"] = float(score * 100.0)
            scores.append(score)
        metrics["mF1"] = float(np.nanmean(scores) * 100.0)
        return metrics
