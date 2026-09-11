_base_ = ["./_base_frequency.py"]

model = dict(
    backbone=dict(
        cloud_adapter_config=dict(
            frequency_ranks=(8, 4, 4), lambda_balance=0.001
        )
    )
)
