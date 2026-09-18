"""Compact, frequent training diagnostics for long iteration-based runs."""

from numbers import Number

import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


def _flatten_numbers(value):
    if isinstance(value, Number):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _flatten_numbers(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _flatten_numbers(item)]
    return []


@HOOKS.register_module()
class TrainingDiagnosticsHook(Hook):
    """Log LR, CUDA memory and aggregate router state at a short interval."""

    priority = "VERY_LOW"

    def __init__(self, interval: int = 50) -> None:
        self.interval = interval

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def after_train_iter(
        self, runner, batch_idx: int, data_batch=None, outputs=None
    ) -> None:
        if not self.every_n_train_iters(runner, self.interval):
            return

        parts = [f"[train-detail] iter={runner.iter + 1}"]
        learning_rates = _flatten_numbers(runner.optim_wrapper.get_lr())
        if learning_rates:
            parts.append(f"lr={learning_rates[0]:.3e}")

        if torch.cuda.is_available():
            gib = 1024**3
            parts.extend(
                (
                    f"cuda_alloc={torch.cuda.memory_allocated() / gib:.2f}GiB",
                    f"cuda_reserved={torch.cuda.memory_reserved() / gib:.2f}GiB",
                    f"cuda_peak={torch.cuda.max_memory_allocated() / gib:.2f}GiB",
                )
            )

        model = self._unwrap(runner.model)
        adapter = getattr(getattr(model, "backbone", None), "cloud_adapter", None)
        if adapter is not None:
            statistics = adapter.get_router_statistics()
            if statistics:
                usage = torch.stack([item["usage"].float().cpu() for item in statistics])
                entropy = torch.stack(
                    [item["entropy"].float().cpu() for item in statistics]
                )
                mean_usage = usage.mean(dim=0).tolist()
                collapsed = int(
                    ((usage > 0.90) | (usage < 0.02)).any(dim=1).sum().item()
                )
                parts.append(
                    "router=" + "/".join(f"{value:.3f}" for value in mean_usage)
                )
                parts.append(f"entropy={entropy.mean().item():.3f}")
                parts.append(f"collapsed_layers={collapsed}/{usage.shape[0]}")

        runner.logger.info(" ".join(parts))
