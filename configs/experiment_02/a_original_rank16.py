_base_ = ["../adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py"]

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val")
    )
)
randomness = dict(seed=42)
model = dict(
    decode_head=dict(
        train_cfg=dict(assigner=dict(type="StableHungarianAssigner"))
    )
)
optim_wrapper = dict(type="AmpOptimWrapper", loss_scale="dynamic")
log_processor = dict(window_size=50, by_epoch=False)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False)
)
custom_hooks = [dict(type="TrainingDiagnosticsHook", interval=50)]
val_evaluator = [
    dict(type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"]),
    dict(type="BoundaryF1Metric", tolerance=3),
]
test_evaluator = val_evaluator
