_base_ = ["../adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py"]

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val")
    )
)
randomness = dict(seed=42)
val_evaluator = [
    dict(type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"]),
    dict(type="BoundaryF1Metric", tolerance=3),
]
test_evaluator = val_evaluator
