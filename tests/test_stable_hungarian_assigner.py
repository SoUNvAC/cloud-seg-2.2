import torch
from mmengine.structures import InstanceData
from mmseg.registry import TASK_UTILS as MMSEG_TASK_UTILS

from cloud_adapter.task_modules import StableHungarianAssigner


def _cost_config():
    return [
        dict(type="ClassificationCost", weight=2.0),
        dict(type="CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
        dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
    ]


def test_assigner_builds_from_mmseg_default_scope():
    assigner = MMSEG_TASK_UTILS.build(
        dict(type="mmdet.StableHungarianAssigner", match_costs=_cost_config())
    )
    assert isinstance(assigner, StableHungarianAssigner)


def test_stable_assigner_handles_non_finite_amp_cost_inputs():
    assigner = StableHungarianAssigner(
        match_costs=_cost_config()
    )
    scores = torch.randn(3, 5, dtype=torch.float16)
    masks = torch.randn(3, 16, dtype=torch.float16)
    scores[0, 0] = torch.inf
    masks[1, 3] = torch.nan
    pred = InstanceData(scores=scores, masks=masks)
    gt = InstanceData(
        labels=torch.tensor([1, 3]),
        masks=torch.randint(0, 2, (2, 16), dtype=torch.long),
    )
    result = assigner.assign(pred, gt)
    assert result.gt_inds.shape == (3,)
    assert (result.gt_inds >= 0).all()
