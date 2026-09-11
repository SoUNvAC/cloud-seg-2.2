_base_ = ["./e_token_balance_844.py"]

model = dict(backbone=dict(cloud_adapter_config=dict(lambda_balance=0.01)))
