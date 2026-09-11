_base_ = ["./_base_frequency.py"]

model = dict(
    backbone=dict(
        cloud_adapter_config=dict(router_mode="token", lambda_balance=0.0)
    )
)
