"""Create the experiment-02 router heatmap and class boxplot from JSONL logs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERTS = ("Spatial", "Low frequency", "High frequency")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="router_stats.jsonl files")
    parser.add_argument("--output-dir", default="work_dirs/experiment_02")
    return parser.parse_args()


def load_last_records(paths):
    records = []
    for path in paths:
        lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
        if not lines:
            raise ValueError(f"router log is empty: {path}")
        records.append(json.loads(lines[-1]))
    return records


def main():
    args = parse_args()
    records = load_last_records(args.logs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    usage = np.asarray(
        [[layer["usage"] for layer in record["layers"]] for record in records]
    ).mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(usage, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set(xlabel="Expert", ylabel="Adapter layer", xticks=range(3))
    ax.set_xticklabels(EXPERTS)
    ax.set_yticks(range(usage.shape[0]))
    ax.set_yticklabels(range(1, usage.shape[0] + 1))
    fig.colorbar(image, ax=ax, label="Mean router weight")
    fig.tight_layout()
    fig.savefig(output_dir / "router_layer_heatmap.png", dpi=200)
    plt.close(fig)

    class_values = {}
    for record in records:
        for layer in record["layers"]:
            for class_name, values in layer.get("class_usage", {}).items():
                class_values.setdefault(class_name, [[], [], []])
                for expert_id, value in enumerate(values):
                    class_values[class_name][expert_id].append(value)
    if not class_values:
        raise ValueError("logs do not contain class_usage values")
    fig, axes = plt.subplots(1, len(class_values), figsize=(4 * len(class_values), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (class_name, values) in zip(axes, class_values.items()):
        ax.boxplot(values, labels=("SP", "LF", "HF"), showfliers=False)
        ax.set_title(class_name)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Per-layer mean router weight")
    fig.tight_layout()
    fig.savefig(output_dir / "router_class_boxplot.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
