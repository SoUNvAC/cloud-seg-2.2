"""Benchmark reference and frequency-routed checkpoints in one environment."""

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch
from mmengine.analysis import get_model_complexity_info
from mmseg.apis import init_model

import cloud_adapter  # noqa: F401 - register project modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--frequency-config", required=True)
    parser.add_argument("--frequency-checkpoint", required=True)
    parser.add_argument("--output", default="work_dirs/experiment_02/efficiency.json")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def measure(config, checkpoint, device, warmup, runs):
    model = init_model(config, checkpoint, device=device).eval()
    adapter = model.backbone.cloud_adapter
    adapter_parameters = sum(parameter.numel() for parameter in adapter.parameters())
    inputs = torch.randn(1, 3, 512, 512, device=device)

    complexity = get_model_complexity_info(
        model, input_shape=(3, 512, 512), show_table=False, show_arch=False
    )
    if isinstance(complexity, dict):
        flops = complexity.get("flops")
    else:
        flops = str(complexity)

    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for step in range(warmup + runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(inputs, mode="tensor")
            end.record()
            torch.cuda.synchronize(device)
            if step >= warmup:
                timings.append(start.elapsed_time(end))
    return {
        "adapter_trainable_params": adapter_parameters,
        "forward_flops": flops,
        "median_latency_ms": median(timings),
        "p90_latency_ms": float(np.percentile(timings, 90)),
        "peak_allocated_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def main():
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The specified FP16 efficiency protocol requires a CUDA GPU")
    result = {
        "status": "complete",
        "hardware": torch.cuda.get_device_name(torch.device(args.device)),
        "reference": measure(
            args.reference_config,
            args.reference_checkpoint,
            args.device,
            args.warmup,
            args.runs,
        ),
        "frequency_routed": measure(
            args.frequency_config,
            args.frequency_checkpoint,
            args.device,
            args.warmup,
            args.runs,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
