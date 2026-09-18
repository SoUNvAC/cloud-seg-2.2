"""Numerically stable Hungarian matching for AMP Mask2Former training."""

import torch
from mmengine.structures import InstanceData
from mmdet.models.task_modules.assigners import HungarianAssigner
from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class StableHungarianAssigner(HungarianAssigner):
    """Run matching costs in FP32 and keep transient AMP overflow feasible.

    Assignment is discrete and does not participate in back-propagation, so
    detached FP32 copies avoid half-precision overflow without changing the
    tensors used by the actual segmentation losses.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def _fp32_copy(instances: InstanceData):
        fields = {}
        device_type = None
        for key, value in instances.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                device_type = value.device.type
                value = value.detach().float()
                # Avoid an extra GPU synchronization just to test finiteness.
                # This is a no-op for normal values and maps invalid matcher
                # entries to a large finite penalty.
                value = torch.nan_to_num(
                    value, nan=1e4, posinf=1e4, neginf=-1e4
                )
            fields[key] = value
        return InstanceData(**fields), device_type

    def assign(self, pred_instances, gt_instances, img_meta=None, **kwargs):
        stable_pred, pred_device = self._fp32_copy(pred_instances)
        stable_gt, gt_device = self._fp32_copy(gt_instances)
        device_type = pred_device or gt_device or "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            return super().assign(
                stable_pred, stable_gt, img_meta=img_meta, **kwargs
            )
