"""Periodic JSONL monitoring for frequency-router behavior."""

import json
import os.path as osp
from pathlib import Path

import torch
import torch.nn.functional as F
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class FrequencyRouterMonitorHook(Hook):
    """Write per-layer usage, entropy, class routing, variance and delta norms."""

    priority = "VERY_LOW"

    def __init__(self, interval: int = 4000, ignore_index: int = 255) -> None:
        self.interval = interval
        self.ignore_index = ignore_index

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def after_train_iter(
        self, runner, batch_idx: int, data_batch=None, outputs=None
    ) -> None:
        if not self.every_n_train_iters(runner, self.interval) or not is_main_process():
            return
        model = self._unwrap(runner.model)
        adapter = getattr(getattr(model, "backbone", None), "cloud_adapter", None)
        if adapter is None:
            return
        layer_stats = adapter.get_router_statistics()
        if not layer_stats:
            return

        targets = []
        for sample in (data_batch or {}).get("data_samples", []):
            gt = getattr(sample, "gt_sem_seg", None)
            if gt is not None:
                targets.append(gt.data)
        target_batch = torch.stack(targets) if targets else None

        class_names = getattr(runner.train_dataloader.dataset, "metainfo", {}).get(
            "classes", ()
        )
        records = []
        for layer, stats in enumerate(layer_stats):
            record = {
                "layer": layer,
                "usage": stats["usage"].float().cpu().tolist(),
                "normalized_entropy": float(stats["entropy"].float().cpu()),
                "token_variance": stats["token_variance"].float().cpu().tolist(),
                "delta_norms": stats["delta_norms"].float().cpu().tolist(),
            }
            if target_batch is not None:
                height, width = stats["spatial_shape"]
                target = F.interpolate(
                    target_batch.float(), size=(height, width), mode="nearest"
                ).long().flatten(1)
                weights = stats["weights"].float().cpu()
                target = target.cpu()
                class_usage = {}
                for class_id, class_name in enumerate(class_names):
                    selected = weights[target == class_id]
                    if selected.numel():
                        class_usage[class_name] = selected.mean(dim=0).tolist()
                record["class_usage"] = class_usage
            records.append(record)

        collapse_count = sum(
            any(value > 0.90 or value < 0.02 for value in record["usage"])
            for record in records
        )
        payload = {
            "iteration": runner.iter + 1,
            "layers": records,
            "collapsed_layers": collapse_count,
        }
        output_path = Path(osp.join(runner.work_dir, "router_stats.jsonl"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if collapse_count > 18:
            runner.logger.warning(
                "Frequency router meets the collapse criterion in "
                f"{collapse_count}/{len(records)} layers."
            )
