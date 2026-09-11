import numpy as np
import torch

from cloud_adapter.metrics.boundary_f1 import BoundaryF1Metric
from cloud_adapter.models.backbones.frequency_routed_adapter import FrequencyDecomposer


def test_frequency_reconstruction_and_backward():
    decomposer = FrequencyDecomposer(cutoff_ratio=0.3)
    for shape in ((2, 64, 32, 32), (2, 64, 16, 16), (1, 64, 24, 32)):
        context = torch.randn(*shape, requires_grad=True)
        spatial, low, high = decomposer(context)
        error = (low + high - spatial).abs().max().item()
        assert error < 1e-5
        (low.square().mean() + high.square().mean()).backward()
        assert context.grad is not None
        assert torch.isfinite(context.grad).all()


def test_boundary_f1_identity_is_one():
    target = torch.tensor(
        [[0, 0, 1, 1], [0, 2, 2, 1], [3, 3, 2, 1], [3, 0, 0, 0]]
    )
    sample = {
        "pred_sem_seg": {"data": target[None]},
        "gt_sem_seg": {"data": target[None]},
    }
    metric = BoundaryF1Metric(tolerance=3)
    metric.dataset_meta = {
        "classes": ("clear", "thick cloud", "thin cloud", "cloud shadow")
    }
    metric.process({}, [sample])
    results = metric.compute_metrics(metric.results)
    assert np.isclose(results["mF1"], 100.0)
