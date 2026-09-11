_base_ = ["../adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py"]

# Validation must not inspect the held-out test split.
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val")
    )
)

model = dict(
    type="FrequencyRoutedEncoderDecoder",
    backbone=dict(
        cloud_adapter_config=dict(
            int_type="frequency_routed",
            rank_dim=None,
            frequency_ranks=(8, 4, 4),
            cutoff_ratio=0.3,
            router_dim=8,
            router_temperature=1.0,
            router_mode="token",
            alpha_init=0.1,
            lambda_balance=0.001,
        )
    ),
)

randomness = dict(seed=42)
custom_hooks = [dict(type="FrequencyRouterMonitorHook", interval=4000)]
val_evaluator = [
    dict(type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"]),
    dict(type="BoundaryF1Metric", tolerance=3),
]
test_evaluator = val_evaluator
